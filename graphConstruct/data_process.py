import pickle
import torch
import numpy as np
import os

from graphConstruct.build_graph import *


def load_graph_data(root_dir, num_omics, node_labels_npy, omics_list, device, trte_idx):

    node_features = []
    adjacency_list_dict = []
    topology = []
    edge_list = []

    num_of_nodes = len(node_labels_npy)

    for i in range(num_omics):
        f = omics_list[i]    
        node_features.append(f)
        edge_list.append(calculateKNNgraphDistanceMatrixStatsWeighted(node_features[i], i, distanceType='euclidean', k=10))
        adjacency_list_dict.append(edgeList2edgeDict(edge_list[i], num_of_nodes))
        topology.append(build_edge_index(adjacency_list_dict[i], num_of_nodes, add_self_edges=True))
        topology[i] = torch.tensor(topology[i], dtype=torch.int32)

    topology_tr = []
    edge_list_tr = []
    node_features_tr = []
    adjacency_list_dict_tr = []
    for i in range(num_omics):
        f = omics_list[i][trte_idx["tr"]]      
        node_features_tr.append(f)

        edge_list_tr.append(calculateKNNgraphDistanceMatrixStatsWeighted(node_features_tr[i], i, distanceType='euclidean', k=10))
        adjacency_list_dict_tr.append(edgeList2edgeDict(edge_list_tr[i], len(trte_idx["tr"])))
        topology_tr.append(build_edge_index(adjacency_list_dict_tr[i], len(trte_idx["tr"]), add_self_edges=True))
        topology_tr[i] = torch.tensor(topology_tr[i], dtype=torch.int32)
    
    topologyFile = os.path.join(root_dir, "topology_trte" + root_dir + ".pkl")
    with open(topologyFile, "wb") as f:
        pickle.dump(topology,f)
        f.close()
    
    topologyFile = os.path.join(root_dir, "topology_tr" + root_dir + ".pkl")
    with open(topologyFile, "wb") as f:
        pickle.dump(topology_tr,f)
        f.close()
    return topology, topology_tr

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

