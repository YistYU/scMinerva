# scMinerva

An unsupervised Single-Cell Multi-omics INtegration method with GCN on hEterogeneous graph utilizing RandomWAlk

## Files
*main.py*: Examples for classification tasks\
*autoEncoder*: Directory contains the GNN autoEncoder\
*graphConstruct*: Directory to build KNN Graph heterogeneous topology with stats one-std based methods using parallel cores\
*randomJump*: Directory to implement a updated random walk - random jump on the given graph.


## Standard Input
*Omics*: .csv file with shape (a,b) where a is the number of sample and b is the number of feature.\
*label*: labels should be indexed start from *0* and be consecutive natural integers. \
*name the omics files*: Files for omics features are supposed to be named as "i.csv" where *i* is an integer to distinguish omics. i.e. "2.csv".\
*name the label files*: Label file is supposed to be named as "labels.csv" under the corresponding dataset directory.
