import torch
import numpy as np
import os

from graphConstruct.build_graph import *

def read_data(data_folder, view_list):
    cuda = False
    num_view = len(view_list)
    data_list = []
    for i in view_list:
        if data_folder == "GSE156478_CITE" or data_folder == "GSE156478_ASAP" : 
            labels = np.loadtxt(os.path.join(data_folder, "labels.csv"), delimiter=',')
            with open(os.path.join(data_folder, str(i)+".csv")) as f:
                ncols = len(f.readline().split(','))
            data = np.loadtxt(os.path.join(data_folder, str(i)+".csv"), delimiter=',', usecols=range(1, ncols), skiprows=1)
            #data = data.T
        if data_folder == "SNARE":
            labels = np.loadtxt(os.path.join(data_folder, "labels.csv"), delimiter=',', usecols=(1))
            with open(os.path.join(data_folder, str(i)+".csv")) as f:
                ncols = len(f.readline().split(','))
            data = np.loadtxt(os.path.join(data_folder, str(i)+".csv"), delimiter=',', usecols=range(1, ncols), skiprows=1)
            data = data.T
        print("Omics", i, "is of shape", data.shape)
        data_list.append(data)  
    return data_list, labels

def load_graph_data(root_dir, num_omics, num_of_samples, omics_list):

    node_features = []
    adjacency_list_dict = []
    topology = []
    edge_list = []


    for i in range(num_omics):
        f = omics_list[i]   
        #m = csr_matrix(f)
        node_features.append(f)
        edge_list.append(calculateKNNgraphDistanceMatrixStatsWeighted(node_features[i], i, k=15))
        print("return edge_list successfully")
        # file_name = os.path.join(root_dir, str(i+1) + '_edge_list.txt')
        # np.savetxt(file_name,edge_list[i])
        adjacency_list_dict.append(edgeList2edgeDict(edge_list[i], num_of_samples))
        topology.append(build_edge_index(adjacency_list_dict[i], num_of_samples, add_self_edges=True))
        topology[i] = torch.tensor(topology[i], dtype=torch.int32)

    # file_name = os.path.join(root_dir, 'topology.pkl')
    # file = open(file_name, 'wb')
    # pickle.dump(topology, file)
    
    return topology



