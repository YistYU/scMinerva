from scipy.spatial import distance
import networkx as nx
import numpy as np
from multiprocessing import Pool
import multiprocessing 
import time


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


#para: measuareName:k:threshold
def calculateKNNgraphDistanceMatrixStatsWeighted(featureMatrix, omics, distanceType='euclidean', k=10, param=None, parallelLimit=0):
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
