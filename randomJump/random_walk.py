# Node2Vec to construct the probability
from unittest.util import sorted_list_difference
import numpy as np
import networkx as nx
import random
import os

class Graph():
	def __init__(self, G, is_directed, p, q, f, node_num):
		self.G = G
		self.is_directed = is_directed

		self.p = p
		self.q = q
		self.f = f

		self.node_num = node_num

	def node2vec_walk(self, walk_length, start_node):
		'''
		Simulate a random walk starting from start node.
		'''
		G = self.G
		alias_nodes = self.alias_nodes
		alias_edges = self.alias_edges
		node_num = self.node_num

		walk = [start_node]

		while len(walk) < walk_length:
			cur = walk[-1]
			cur_nbrs = sorted(G.neighbors(cur))
			if len(cur_nbrs) > 0:
				if len(walk) == 1:
					#print(len(alias_nodes[cur]))
					walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
				else:
					prev = walk[-2]
					next = cur_nbrs[alias_draw(alias_edges[(prev, cur)][0], 
						alias_edges[(prev, cur)][1])]
					walk.append(next)
			else:
				break

		return walk

	def simulate_walks(self, num_walks, walk_length, args):
		'''
		Repeatedly simulate random walks from each node.
		'''
		G = self.G
		walks = []
		nodes = list(G.nodes())
		#print ("Walk iteration:")
		for walk_iter in range(num_walks):
			#print (str(walk_iter+1), '/', str(num_walks))
			random.shuffle(nodes)
			for node in nodes:
				walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))

		return walks


	def get_alias_edge(self, src, dst):
		'''
		Get the alias edge setup lists for a given edge.
		'''
		G = self.G

		p = self.p
		q = self.q
		f = self.f
		node_num = self.node_num
		unnormalized_probs = []

		dst_graph = (int)(dst / node_num)
		src_graph = (int)(src / node_num)
		#print(dst_graph, src_graph)

		if dst_graph == src_graph:
			for dst_nbr in sorted(G.neighbors(dst)):
				if dst_nbr == src:
					unnormalized_probs.append(G[dst][dst_nbr]['weight']/p)
				elif G.has_edge(dst_nbr, src) and dst_nbr <= node_num:
					unnormalized_probs.append(G[dst][dst_nbr]['weight'])
				elif (not G.has_edge(dst_nbr, src)) and dst_nbr <= node_num:
					unnormalized_probs.append(G[dst][dst_nbr]['weight']/q)
				elif G.has_edge(dst_nbr, src) and dst_nbr > node_num:
					unnormalized_probs.append(G[dst][dst_nbr]['weight']/f)
				elif (not G.has_edge(dst_nbr, src)) and dst_nbr > node_num:
					unnormalized_probs.append(G[dst][dst_nbr]['weight']/( q* f))
				else:
					print("Report Error")
		
		if dst_graph != src_graph:
			for dst_nbr in sorted(G.neighbors(dst)):
				if dst_nbr == src:
					unnormalized_probs.append(G[dst][dst_nbr]['weight']/(p * f))
				elif G.has_edge(dst_nbr, src) and dst_nbr <= node_num:
					unnormalized_probs.append(G[dst][dst_nbr]['weight']/f)
				elif (not G.has_edge(dst_nbr, src)) and dst_nbr <= node_num:
					unnormalized_probs.append(G[dst][dst_nbr]['weight']/(q * f))
				elif G.has_edge(dst_nbr, src) and dst_nbr > node_num:
					unnormalized_probs.append(G[dst][dst_nbr]['weight'])
				elif (not G.has_edge(dst_nbr, src)) and dst_nbr > node_num:
					unnormalized_probs.append(G[dst][dst_nbr]['weight']/( q))
				else:
					print("Report Error")

		self.G = G
		norm_const = sum(unnormalized_probs)
		normalized_probs =  [float(u_prob) /norm_const for u_prob in unnormalized_probs]
		#print(normalized_probs)
	
		return alias_setup(normalized_probs)

	def preprocess_transition_probs(self, data_folder):
		'''
		Preprocessing of transition probabilities for guiding the random walks.
		'''
		G = self.G
		is_directed = self.is_directed
		node_num = self.node_num
		edge_list = []
		alias_nodes = {}
		for node in G.nodes():	
			unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
			norm_const = sum(unnormalized_probs)
			normalized_probs = [float(u_prob)/norm_const for u_prob in unnormalized_probs]

			for i in range(0, len(sorted(G.neighbors(node)))):
				edge_list.append((node, sorted(G.neighbors(node))[i], normalized_probs[i]))
			# for prob in normalized_probs:
			# 	prob = prob.detach().numpy()
			ar = np.zeros(len(normalized_probs))
			for e in range(len(normalized_probs)):
				ar[e] = normalized_probs[e]
			normalized_probs = np.array(ar)
			alias_nodes[node] = alias_setup(normalized_probs)
		alias_edges = {}
		triads = {}

		if not is_directed:
			for edge in G.edges():
				alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
		else:
			for edge in G.edges():
				alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
				alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

		self.alias_nodes = alias_nodes
		self.alias_edges = alias_edges

		return edge_list

def alias_setup(probs):
	'''
	Compute utility lists for non-uniform sampling from discrete distributions.
	Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
	for details
	'''
	K = len(probs)
	#print(K)
	q = np.zeros(K)
	J = np.zeros(K, dtype=np.int)
	
	#print(probs.shape)
	smaller = []
	larger = []
	for kk, prob in enumerate(probs):
		q[kk] = K*prob
		if q[kk] < 1.0:
			smaller.append(kk)
		else:
			larger.append(kk)

	while len(smaller) > 0 and len(larger) > 0:
		small = smaller.pop()
		large = larger.pop()

		J[small] = large
		q[large] = q[large] + q[small] - 1.0
		if q[large] < 1.0:
			smaller.append(large)
		else:
			larger.append(large)

	return J, q

def alias_draw(J, q):
	'''
	Draw sample from a non-uniform discrete distribution using alias sampling.
	'''
	K = len(J)

	kk = int(np.floor(np.random.rand()*K))
	if np.random.rand() < q[kk]:
		return kk
	else:
		return J[kk]

def get_node_normalized_prob(Graph, node, weight, node_num):
	
	w = weight
	unnormalized_probs = [[],[]]
	normalized_probs = [[],[]]
	#print(Graph[0].nodes)
	res = []
	node = node & (node_num)
	for idx, G in enumerate(Graph):
		#print(node, idx)
		try: 
			unnormalized_probs[idx] = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
			norm_const = sum(unnormalized_probs[idx])
			normalized_probs[idx] =  [float(u_prob) * w[idx]/norm_const for u_prob in unnormalized_probs[idx]]
			node += node_num + 1
		except nx.exception.NetworkXError:
			print("Error message")
			print(node)
			print(idx)
		
		else:
			continue
		
	res = normalized_probs[0] + normalized_probs[1]

	return res