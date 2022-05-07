from asyncio import queues
import torch
import numpy as np
import networkx as nx
import scanpy as sc
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
import math
import pandas as pd
import os
import pickle
import gensim
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split

import randomJump.random_walk as random_walk
import randomJump.jump_weight_models as jump_weight_models
from randomJump.evaluate import lossFunc
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

def recon_Graph(weight, args, edge_list, node_num, data_folder):

	
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
				omics_gap = (int) (((edge[0] - edge[1]) / fold) - 1)
				G[edge[0]][edge[1]]['weight'] = weight[(int)(edge[0] % fold)][omics_gap * 2].detach().numpy()+args.delta
				G[edge[1]][edge[0]]['weight'] = weight[(int)(edge[0] % fold)][omics_gap * 2 + 1].detach().numpy()+args.delta
			if edge[0] < edge[1]:
				continue
			if edge[0] == edge[1]:
				G[edge[0]][edge[1]]['weight'] = 1
		else:
			G[edge[0]][edge[1]]['weight'] = 1

	# print(G.edges())
	return G

def learn_embeddings(args, walks, node_num):
	'''
	Learn embeddings by optimizing.
	'''
	model_heter = Word2Vec(list(walks), vector_size=args.dimensions, window=args.window_size, min_count=0, sg=0, workers=args.workers, epochs=args.iter)

	walks_regen = []
	walks_keep = walks
	for walk in walks_keep:
		for i in range(len(walk)):
			omics_fold = math.floor(walk[i]/ (node_num ))
			# print(omics_fold)
			walk[i] = walk[i] - (node_num ) * (omics_fold)
		walks_regen.append(walk)
	model_sample = Word2Vec(list(walks_regen), vector_size=args.dimensions, window=args.window_size, min_count=0, sg=0, workers=args.workers, epochs=args.iter)
	return model_heter.wv, model_sample.wv, walks_regen
	

def train_RJ(args, epoch, version, mu, p, weight, node_num, G, model_dict, optim_dict, topology, num_class, data_folder):
	if epoch >= 1:
		weight = weight.clone().detach().requires_grad_(True)
		weight = weight.to(torch.float32)
	recon_G = recon_Graph(weight, args, topology, node_num, data_folder)
	G = random_walk.Graph(recon_G, True, args.p, args.q, args.z, node_num)
	full_edge_list = G.preprocess_transition_probs(data_folder)
	walks = G.simulate_walks(args.num_walks, args.walk_length, args)


	wv_heter, wv_sample, walks_regenerate = learn_embeddings(args, walks, node_num)
	df = pd.DataFrame(data=wv_heter.vectors)
	df.index = wv_heter.key_to_index.keys()
	DATA = np.array(df.sort_index())
	print(DATA.shape)
	embedding_input = torch.from_numpy(DATA) 
	edges = recon_G.edges.data() 
	edge_list = []
	for item in edges:
		edge_list.append((item[0], item[1]))
	edges = np.array(edge_list)
	edges = edges.T
	edges = torch.from_numpy(edges)
	weight = model_dict(embedding_input, edges)
	
	kmeans = KMeans(num_class * (len(topology)+1), n_init=5)
	y_pred = kmeans.fit_predict(DATA)
	DATA = torch.from_numpy(DATA)
	
	if epoch == 0:
		mu = Parameter(torch.Tensor(num_class, DATA.shape[1]))
		features=pd.DataFrame(DATA.detach().numpy(),index=np.arange(0,DATA.shape[0]))
		Group=pd.Series(y_pred,index=np.arange(0,features.shape[0]),name="Group")
		Mergefeature=pd.concat([features,Group],axis=1)
		cluster_centers=np.asarray(Mergefeature.groupby("Group").mean())
		mu.data.copy_(torch.Tensor(cluster_centers))
	if epoch % 4 == 0:
		q = distribution(DATA, num_class, mu)
		p = target_distribution(q).data
	q = distribution(DATA, num_class, mu)
	q.requires_grad_()
	cost = loss_function(p,q)
	print("For epoch", epoch, "the cost is", cost.detach().numpy())
	jump_weight_models.train_para_vec(args, weight, model_dict, cost, optim_dict)
	df = pd.DataFrame(data=wv_sample.vectors)
	df.index = wv_sample.key_to_index.keys()
	DATA = np.array(df.sort_index())

	return weight, DATA, mu, p, full_edge_list, walks_regenerate

def track_RJ(DATA,label, test_list):
	acc = 0
	for item in test_list:
		print("test size is ", item)
		DATA_tr, DATA_te, tr_LABEL, te_LABEL = train_test_split(DATA, label, test_size=item, random_state=42)
		clf = KNeighborsClassifier(n_neighbors=8)
		clf.fit(DATA_tr, tr_LABEL)
		L_pred = clf.predict(DATA_te)
		if te_LABEL.max() == 1:
			print("ACC: {:.3f}".format(accuracy_score(te_LABEL, L_pred, normalize=True)))
			print("F1-score: {:.3f}".format(f1_score(te_LABEL, L_pred)))
			print("ARI: {:.3f}".format(adjusted_rand_score(te_LABEL, L_pred)))
		else:
			print("ACC: {:.3f}".format(accuracy_score(te_LABEL, L_pred, normalize=True)))
			print("F1-weighted: {:.3f}".format(f1_score(te_LABEL, L_pred, average='weighted')))
			print("ARI: {:.3f}".format(adjusted_rand_score(te_LABEL, L_pred)))
		print(" ")



def RJ(args, topology, view_num, label, test_list, data_folder, num_class, version):

	node_num = len(label)
	nx_G = read_graph(args, topology, node_num)
	args.nn_dim = (int)(view_num * (view_num - 1))
	model_dict, optim_dict = jump_weight_models.initialize_train_para_vec(args, num_class)
	epoch = args.Jump_epochs
	weight = torch.ones(len(label), view_num * (view_num-1)) * 100
	mu = 0
	p = 0
	for i in range(0, epoch):
		print("Epoch", i)
		weight, DATA, mu, p, full_edge_list, walks_regenerate= train_RJ(args, i, version, mu, p, weight, node_num, nx_G, model_dict, optim_dict, topology, num_class, data_folder)
		if i%4 == 0:
			track_RJ(DATA, label, test_list)
		
		






