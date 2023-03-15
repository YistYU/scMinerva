from scipy.spatial import distance
import networkx as nx
import numpy as np
from multiprocessing import Pool
import multiprocessing 
import time
from sklearn.ensemble import IsolationForest

# kernelDistance
def kernelDistance(distance,delta=1.0):
    '''
    Calculate kernel distance
    '''
    kdist = np.exp(-distance/2*delta**2)
    return kdist


class FindKParallel():
    '''
    A class to find K parallel
    '''
    def __init__(self,featureMatrix,distanceType,k):
        self.featureMatrix = featureMatrix
        self.distanceType = distanceType
        self.k = k

    def vecfindK(self,i):
        '''
        Find topK in paral
        '''
        edgeList_t=[]
        tmp=self.featureMatrix[i,:].reshape(1,-1)
        distMat = distance.cdist(tmp,self.featureMatrix, self.distanceType)
        # print('#'+str(distMat))
        res = distMat.argsort()[:self.k+1]
        # print('!'+str(res))
        tmpdist = distMat[0,res[0][1:self.k+1]]
        # print('@'+str(tmpdist))
        boundary = np.mean(tmpdist)+np.std(tmpdist)
        # print('&'+str(boundary))
        for j in np.arange(1,self.k+1):
            # TODO: check, only exclude large outliners
            # if (distMat[0,res[0][j]]<=mean+std) and (distMat[0,res[0][j]]>=mean-std):
            if distMat[0,res[0][j]]<=boundary:
                weight = kernelDistance(distMat[0,res[0][j]])
                edgeList_t.append((i,res[0][j],weight))
        # print('%'+str(len(edgeList_t)))
        return edgeList_t
    
    def work(self):
        return Pool().map(self.vecfindK, range(self.featureMatrix.shape[0]))

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

def calculateKNNgraphDistanceMatrixML(featureMatrix, distanceType='mahalanobis', k=10, param=None):
    r"""
    Thresholdgraph: KNN Graph with Machine Learning based methods

    IsolationForest
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html#sklearn.ensemble.IsolationForest 
    """      
    # featureMatrix = featureMatrix.toarray()
    distMat = distance.cdist(featureMatrix,featureMatrix, distanceType)
    edgeList=[]
    # parallel: n_jobs=-1 for using all processors
    print("Inside to do isolation forest")
    clf = IsolationForest( contamination= 'auto', n_jobs=-1)

    for w in np.arange(distMat.shape[0]):
        if w % 100 == 0:
            print ("w=", w)
        res = distMat[w,:].argsort()[:k+1]
        preds = clf.fit_predict(featureMatrix[res,:])       
        for j in np.arange(1,k+1):
            # weight = 1.0
            if preds[j]==-1:
                weight = 0.0
            else:
                weight = 1.0
            #preds[j]==-1 means outliner, 1 is what we want
            edgeList.append((w,res[j],weight))
        if w == distMat.shape[0] - 1:
            break
    return edgeList

#para: measuareName:k:threshold
def calculateKNNgraphDistanceMatrixStatsWeighted(featureMatrix, omics, distanceType='Euclidean', k=8, param=None, parallelLimit=0):
    r"""
    Thresholdgraph: KNN Graph with stats one-std based methods using parallel cores
    """       
    edgeListWeighted=[]
    # Get number of availble cores
    USE_CORES = 0 
    NUM_CORES = multiprocessing.cpu_count()
    # if no limit, use all cores
    if parallelLimit == 0:
        USE_CORES = NUM_CORES
    # if limit < cores, use limit number
    elif parallelLimit < NUM_CORES:
        USE_CORES = parallelLimit
    # if limit is not valid, use all cores
    else:
        USE_CORES = NUM_CORES

    t= time.time()
    #Use number of cpus for top-K finding
    with Pool(USE_CORES) as p:
        # edgeListT = p.map(vecfindK, range(featureMatrix.shape[0]))
        edgeListT = FindKParallel(featureMatrix, distanceType, k).work()

    t1=time.time()
    flatten = lambda l: [item for sublist in l for item in sublist]   
    t2=time.time()
    edgeListWeighted = flatten(edgeListT)    
       
    return edgeListWeighted


# edgeList to edgeDict
def edgeList2edgeDict(edgeList, nodesize):
    graphdict={}
    tdict={}

    for edge in edgeList:
        end1 = edge[0]
        end2 = edge[1]
        tdict[end1]=""
        tdict[end2]=""
        if end1 in graphdict:
            tmplist = graphdict[end1]
        else:
            tmplist = []
        tmplist.append(end2)
        graphdict[end1]= tmplist

    #check and get full matrix
    for i in range(nodesize):
        if i not in tdict:
            graphdict[i]=[]

    return graphdict
