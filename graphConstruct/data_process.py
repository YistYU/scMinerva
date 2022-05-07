import pickle
import torch
import numpy as np
import os

from graphConstruct.build_graph import *



def load_graph_data(root_dir, num_omics, num_of_nodes, omics_list):

    node_features = []
    adjacency_list_dict = []
    topology = []
    edge_list = []


    for i in range(num_omics):
        f = omics_list[i]   
        #m = csr_matrix(f)
        node_features.append(f)
        edge_list.append(calculateKNNgraphDistanceMatrixStatsWeighted(node_features[i], i, distanceType='euclidean', k=15))
        print("return edge_list successfully")
        file_name = os.path.join(root_dir, str(i+1) + '_edge_list.txt')
        np.savetxt(file_name,edge_list[i])
        adjacency_list_dict.append(edgeList2edgeDict(edge_list[i], num_of_nodes))
        topology.append(build_edge_index(adjacency_list_dict[i], num_of_nodes, add_self_edges=True))
        topology[i] = torch.tensor(topology[i], dtype=torch.int32)

    file_name = os.path.join(root_dir, 'topology.pkl')
    file = open(file_name, 'wb')
    pickle.dump(topology, file)
    
    return topology


def build_edge_index(adjacency_list_dict, num_of_nodes, add_self_edges=True):
    source_nodes_ids, target_nodes_ids = [], []
    seen_edges = set()

    for src_node, neighboring_nodes in adjacency_list_dict.items():
        for trg_node in neighboring_nodes:
            # if this edge hasn't been seen so far we add it to the edge index (coalescing - removing duplicates)
            if (src_node, trg_node) not in seen_edges:  # it'd be easy to explicitly remove self-edges (Cora has none..)
                source_nodes_ids.append(src_node)
                target_nodes_ids.append(trg_node)

                seen_edges.add((src_node, trg_node))

    if add_self_edges:
        source_nodes_ids.extend(np.arange(num_of_nodes))
        target_nodes_ids.extend(np.arange(num_of_nodes))

    # shape = (2, E), where E is the number of edges in the graph
    edge_index = np.row_stack((source_nodes_ids, target_nodes_ids))

    return edge_index

