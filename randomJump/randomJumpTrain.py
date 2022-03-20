import torch
import numpy as np
import networkx as nx
from gensim.models import Word2Vec
import math
import pandas as pd
import gensim
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import randomJump.random_walk as random_walk
import randomJump.jump_weight_models as jump_weight_models
from randomJump.evaluate import lossFunc

gensim.models.word2vec.FAST_VERSION == 1
cuda = True if torch.cuda.is_available() else False

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
		fold = i * (node_num + 1)
		G.append(nx.Graph())
		
		for ct in range(len(edge_list[i][1])):
			G[i].add_edge(edge_list[i][0][ct].item() + fold, edge_list[i][1][ct].item() + fold)
		for edge in G[i].edges():
			G[i][edge[0]][edge[1]]['weight'] = 1
	return G

def recon_Graph(nx_G, weight, args, edge_list, node_num):

	
	G = nx.DiGraph()
	# add all the edges from the original graph
	for i in range(len(edge_list)):
		fold = i * (node_num + 1)
		edge_list[i] = edge_list[i].cpu()
		for ct in range(len(edge_list[i][1])):
			G.add_edge(edge_list[i][0][ct].item() + fold, edge_list[i][1][ct].item() + fold)
			G.add_edge(edge_list[i][1][ct].item() + fold, edge_list[i][0][ct].item() + fold)

	# add all the edges which link different omics
	for i in range(1, len(edge_list)):
		fold_i = i * (node_num + 1)
		for j in range(0,i):
			fold_j = j * (node_num + 1)
			for node in range(node_num):
				G.add_edge(node + fold_i, node + fold_j)
				G.add_edge(node + fold_j, node + fold_i)

	# Plug in the trained weight
	fold = node_num + 1
	for edge in G.edges():
		gap = (edge[0] - edge[1]) % fold
		if gap == 0:
			if edge[0] > edge[1]:
				omics_gap = (int) (((edge[0] - edge[1]) / fold) - 1)
				G[edge[0]][edge[1]]['weight'] = weight[(int)(edge[0] % fold)][omics_gap * 2].detach().numpy()
				G[edge[1]][edge[0]]['weight'] = weight[(int)(edge[0] % fold)][omics_gap * 2 + 1].detach().numpy()+args.delta
			if edge[0] < edge[1]:
				continue
			if edge[0] == edge[1]:
				G[edge[0]][edge[1]]['weight'] = 1
		else:
			G[edge[0]][edge[1]]['weight'] = 1
	return G

def learn_embeddings(args, walks, node_num):
	'''
	Learn embeddings by optimizing.
	'''
	for walk in walks:
		for i in range(len(walk)):
			omics_fold = math.floor(walk[i]/ (node_num + 1))
			walk[i] = walk[i] - (node_num + 1) * (omics_fold)
	model = Word2Vec(list(walks), vector_size=args.dimensions, window=args.window_size, min_count=0, sg=0, workers=args.workers, epochs=args.iter)
	
	return model.wv

def train_RJ(args, epoch, weight, G, model_dict, criterion, optim_dict, gt_LABEL, topology, num_class, data_folder):

	if epoch >= 1:
		weight = weight.clone().detach().requires_grad_(True)
		weight = weight.to(torch.float32)
		weight = model_dict(args, weight)
	node_num = len(gt_LABEL)

	recon_G = recon_Graph(G, weight, args, topology, node_num)
	G = random_walk.Graph(recon_G, True, args.p, args.q, args.z, node_num)
	G.preprocess_transition_probs()
	walks = G.simulate_walks(args.num_walks, args.walk_length, args)
	wv = learn_embeddings(args, walks, node_num)
	df = pd.DataFrame(data=wv.vectors)
	df.index = wv.key_to_index.keys()
	DATA = np.array(df.sort_index())

	gt, pred= lossFunc(args, DATA, gt_LABEL)
	cost = criterion(pred, gt)
	jump_weight_models.train_para_vec(args, weight, model_dict, cost, optim_dict)

	return weight


	
def test_RJ(args, weight, G, model_dict, trte_idx, label, topology, num_class, data_folder):
	weight = model_dict(args, weight)
	node_num = len(trte_idx['tr']) + len(trte_idx['te'])
	recon_G = recon_Graph(G, weight, args, topology, node_num)
	G = random_walk.Graph(recon_G, True, args.p, args.q, args.z, node_num)
	G.preprocess_transition_probs()
	walks = G.simulate_walks(args.num_walks, args.walk_length, args)
	wv = learn_embeddings(args, walks, node_num)
	df = pd.DataFrame(data=wv.vectors)
	df.index = wv.key_to_index.keys()
	DATA = np.array(df.sort_index())
	DATA_te = DATA[trte_idx["te"]]
	DATA_tr = DATA[trte_idx["tr"]]
	scaler = StandardScaler()
	scaler.fit(DATA_tr)
	scaler.fit(DATA_te)
	clf = KNeighborsClassifier(n_neighbors=8)
	tr_LABEL = label[trte_idx["tr"]]
	gt_LABEL = label[trte_idx['te']]
	clf.fit(DATA[trte_idx["tr"]], tr_LABEL)
	DATA_te = scaler.transform(DATA_te)
	L_pred = clf.predict(DATA_te)
	if gt_LABEL.max() == 1:
		print("ACC: {:.3f}".format(accuracy_score(gt_LABEL, L_pred, normalize=True)))
		print("F1-score: {:.3f}".format(f1_score(gt_LABEL, L_pred)))
		print("AUC: {:.3f}".format(roc_auc_score(gt_LABEL, L_pred)))
	else:
		print("ACC: {:.3f}".format(accuracy_score(gt_LABEL, L_pred, normalize=True)))
		print("F1-weighted: {:.3f}".format(f1_score(gt_LABEL, L_pred, average='weighted')))
		print("F1-macro: {:.3f}".format(f1_score(gt_LABEL, L_pred, average='macro')))

def train_test_RJ(args, topology, topology_tr, view_num, trte_idx, label, data_folder, sample_weight, num_class):

	node_num_tr = len(trte_idx["tr"])
	node_num_trte = len(label)
	nx_G_tr = read_graph(args, topology_tr, node_num_tr)
	nx_G_trte = read_graph(args, topology, node_num_trte)
	args.nn_dim = (int)(view_num * (view_num - 1))
	model_dict, criterion, optim_dict = jump_weight_models.initialize_train_para_vec(args, num_class)
	epoch = args.Jump_epochs
	tr_label = np.array(trte_idx["tr"])
	weight_tr = torch.ones(len(tr_label), view_num * (view_num-1))
	weight_trte = torch.ones(len(label), view_num * (view_num-1))
	for i in range(0, epoch):
		print("Epoch", i)
		weight_tr = train_RJ(args, i, weight_tr, nx_G_tr, model_dict, criterion, optim_dict, label[trte_idx["tr"]], topology_tr, num_class, data_folder)
		test_RJ(args, weight_trte, nx_G_trte, model_dict, trte_idx, label, topology, num_class, data_folder)




