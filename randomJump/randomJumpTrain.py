from re import T
import torch
import numpy as np
import networkx as nx
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
import math
import pandas as pd
import gensim

from graphConstruct.build_graph import *
import randomJump.random_walk as random_walk
import randomJump.jump_weight_models as jump_weight_models
from torch.nn.parameter import Parameter

gensim.models.word2vec.FAST_VERSION == 1
cuda = True if torch.cuda.is_available() else False

def target_distribution(q):
	p = q**2 / torch.sum(q, dim=0)
	p = p / torch.sum(p, dim=1, keepdim=True)
	return p

def distribution(x, num_class, mu):
	alpha = 0.05
	q = 1.0 / ((1.0 + torch.sum((x.unsqueeze(1) - mu)**2, dim=2) / alpha) + 1e-6)
	q = q**(alpha+1.0)/2.0
	q = q / torch.sum(q, dim=1, keepdim=True)
	return q

def loss_function(p, q):
	def kld(target, pred):
		return torch.mean(torch.sum(target*torch.log(target/(pred+1e-6)), dim=1))
	loss = kld(p, q)
	return loss


def read_graph(args, edge_list, node_num):
	'''
	Read the input graph from different omics
	'''
	
	G = []
	for i in range(len(edge_list)):
		edge_list[i] = edge_list[i].cpu()
	if max(edge_list[0][0]) != max(edge_list[1][0]):
		print("Graph Nodes Inconsistent.")
		exit()
	
	for i in range(len(edge_list)):
		fold = i * node_num 
		G.append(nx.Graph())
		
		for ct in range(len(edge_list[i][1])):
			G[i].add_edge(edge_list[i][0][ct].item() + fold, edge_list[i][1][ct].item() + fold)
		for edge in G[i].edges():
			G[i][edge[0]][edge[1]]['weight'] = 1
	return G

def Relu(inX):
    return np.maximum(0,inX)


def recon_Graph(weight, args, edge_list, node_num, data_folder, sample=False):

	
	G = nx.DiGraph()
	# add all the edges from the original graph
	for i in range(len(edge_list)):
		fold = i * node_num 
		edge_list[i] = edge_list[i].cpu()
		for ct in range(len(edge_list[i][1])):
			G.add_edge(edge_list[i][0][ct].item() + fold, edge_list[i][1][ct].item() + fold)
			G.add_edge(edge_list[i][1][ct].item() + fold, edge_list[i][0][ct].item() + fold)

	# add all the edges which link different omics
	for i in range(1, len(edge_list)):
		fold_i = i * node_num
		for j in range(0,i):
			fold_j = j * node_num
			for node in range(node_num):
				G.add_edge(node + fold_i, node + fold_j)
				G.add_edge(node + fold_j, node + fold_i)

	# Plug in the trained weight
	fold = node_num
	for edge in G.edges():
		gap = (edge[0] - edge[1]) % fold
		if gap == 0:
			if edge[0] > edge[1]:
				G[edge[0]][edge[1]]['weight'] = weight[(edge[0])][edge[1]].detach().numpy()+args.delta
				G[edge[1]][edge[0]]['weight'] = weight[(edge[1])][edge[0]].detach().numpy()+args.delta
			if edge[0] < edge[1]:
				continue
			if edge[0] == edge[1]:
				G[edge[0]][edge[1]]['weight'] = 1
		else:
			G[edge[0]][edge[1]]['weight'] = 1

	# print(G.edges())
	return G


def learn_embeddings(args, walks, node_num, sample=False):
	'''
	Learn embeddings by optimizing.
	'''
	if sample is False:
		walks_regen = []
		emb = Word2Vec(list(walks), vector_size=args.dimensions, window=args.window_size, min_count=0, sg=0, workers=args.workers, epochs=args.iter)
	else:
		walks_regen = []
		for walk in walks:
			for i in range(len(walk)):
				omics_fold = math.floor(walk[i]/ (node_num ))
				# print(omics_fold)
				walk[i] = walk[i] - (node_num ) * (omics_fold)
			walks_regen.append(walk)
		emb = Word2Vec(list(walks_regen), vector_size=args.dimensions, window=args.window_size, min_count=0, sg=0, workers=args.workers, epochs=args.iter)
	return emb.wv
	

def train_RJ(args, num_of_sample, G, topology, num_class, data_folder):
	mu = 0
	p = 0
	model_dict, optim_dict = jump_weight_models.initialize_train_para_vec(args, args.num_class)
	num_of_node = (int)(num_of_sample * args.num_omics)
	weight = torch.ones(num_of_node, num_of_node) 

	print("\n ----Start omics2vec on heterogeneous graph----\n ")
	recon_G = recon_Graph(weight, args, topology, num_of_sample, data_folder)
	G = random_walk.Graph(recon_G, True, args.p, args.q, args.z, num_of_sample)
	full_edge_list = G.preprocess_transition_probs(data_folder)
	walks = G.simulate_walks(args.num_walks, args.walk_length, args)
	wv_nodes = learn_embeddings(args, walks, num_of_sample)
	print("\n ----Finish omics2vec on heterogeneous graph----\n ")

	df = pd.DataFrame(data=wv_nodes.vectors)
	df.index = wv_nodes.key_to_index.keys()
	DATA = np.array(df.sort_index())
	embedding_input = torch.from_numpy(DATA) 
	edges = recon_G.edges.data() 
	edge_list = []
	for item in edges:
		edge_list.append((item[0], item[1]))
	edges = np.array(edge_list)
	edges = edges.T
	edges = torch.from_numpy(edges)
	weight = model_dict(embedding_input, edges)
	kmeans = KMeans(num_class , n_init=42)
	y_pred = kmeans.fit_predict(DATA)
	DATA = torch.from_numpy(DATA)
	inner_epoch=100
	cls_dec = np.int(num_class) * np.int(len(topology)) 
	x = model_dict(DATA, edges)
	kmeans = KMeans(cls_dec, n_init=cls_dec, random_state=42)
	x_copy = np.copy(x.cpu().detach().numpy())
	dec_tol = 5e-2
	y_pred_last = kmeans.fit_predict(x_copy)
	
	print("\n ----Start GCN training----\n ")
	for counter in range(inner_epoch):
		x = model_dict(DATA, edges)
		x_copy = np.copy(x.cpu().detach().numpy())
		y_pred = kmeans.fit_predict(x_copy)
		if counter % 6 == 0:
			mu = Parameter(torch.Tensor(cls_dec, x.shape[1]))
			features=pd.DataFrame(x,index=np.arange(0,x.shape[0]))
			Group=pd.Series(y_pred,index=np.arange(0,features.shape[0]),name="Group")
			Mergefeature=pd.concat([features,Group],axis=1)
			cluster_centers=np.asarray(Mergefeature.groupby("Group").mean())
			mu.data.copy_(torch.Tensor(cluster_centers))
			q = distribution(x, cls_dec, mu)
			p = target_distribution(q).data
			y_pred = p.cpu().numpy().argmax(1)
			delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
			y_pred_last = np.copy(y_pred)
			if delta_label < dec_tol:
				print('delta_label {:.4}'.format(delta_label), '< tol', dec_tol)
				print('Reached tolerance threshold. Stopping training.')
				break
		else:
			q = distribution(x, cls_dec, mu)
			q.requires_grad_(True)
		cost = loss_function(p,q)
		cost.backward()
		optim_dict.step()
	x = model_dict(DATA, edges)
	x = x.cpu().detach().numpy()
	weight = np.ones((len(x), len(x)))
	for i in range(len(x)):
		for j in range(len(x)):
			weight[i][j] = np.dot(x[i], x[j])
	weight = Relu(weight)
	weight = torch.from_numpy(weight)

	print("\n ----Run omics2vec on trained graph ----\n ")
	recon_G = recon_Graph(weight, args, topology, num_of_sample, data_folder)
	G = random_walk.Graph(recon_G, True, args.p, args.q, args.z, num_of_sample)
	full_edge_list = G.preprocess_transition_probs(data_folder)
	walks = G.simulate_walks(args.num_walks, args.walk_length, args)
	wv_samples= learn_embeddings(args, walks, num_of_sample, sample=True)
	df = pd.DataFrame(data=wv_samples.vectors)
	df.index = wv_samples.key_to_index.keys()
	DATA = np.array(df.sort_index())
	print("\n ----Finish omics2vec on trained graph ----\n ")

	return DATA