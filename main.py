import torch
cuda = True if torch.cuda.is_available() else False

import argparse
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import KNeighborsClassifier
from randomJump.randomJumpTrain import *
from graphConstruct.data_process import *

def parse_args():
	'''
	Parses the RandomJump arguments.
	'''
	parser = argparse.ArgumentParser(description="Run RandomJump.")
        
	parser.add_argument('--data_folder', type=str, default='SNARE',
	                    help='The input dataset')
	parser.add_argument('--num_omics', type=int, default=2,
	                    help='The amount of omics')
	parser.add_argument('--num_class', type=int, default=4,
	                    help='The number of cell types/stages for classification')
	parser.add_argument('--labeled_ratio', type=float, default=0.05,
	                    help='The percentage of labeled data for fine-tuning')
                        
	# Parameters for the generation of walks
	parser.add_argument('--dimensions', type=int, default=64,
	                    help='Number of dimensions. Default is 64.')

	parser.add_argument('--walk-length', type=int, default=42,
	                    help='Length of walk per source. Default is 42.')

	parser.add_argument('--num-walks', type=int, default=20,
	                    help='Number of walks per source. Default is 10.')

	parser.add_argument('--window-size', type=int, default=10,
                    	help='Context size for optimization. Default is 10.')

	parser.add_argument('--iter', default=1, type=int,
                        help='Number of epochs in SGD')

	parser.add_argument('--workers', type=int, default=120,
	                    help='Number of parallel workers. Default is 8.')

	#Jumping probability hyperparamters
	parser.add_argument('--p', type=float, default=1,
	                    help='Return hyperparameter. Default is 1.')

	parser.add_argument('--q', type=float, default=2,
	                    help='Inout hyperparameter. Default is 1.')

	parser.add_argument('--z', type=float, default=0.5,
	                    help='Jump hyperparameter. Default is 1.')

	parser.add_argument('--delta', type=float, default=1e-3,
	                    help='Default Jump hyper-probablity. Default is 1e-3.')

	#For the training of model for jump parameters 
	parser.add_argument('--Jump_epochs', type=int, default=20,
	                    help='The number of epoch for jump parameter training')
	
	parser.add_argument('--Jump_lr', type=float, default=0.0001,
	                    help='The learning rate for jump parameter training')

	parser.add_argument('--Jump_dropout', type=float, default=0.8,
	                    help='The dropout for jump parameter training')

	parser.add_argument('--Jump_bias', action='store_true', default=True,
	                    help='Boolean specifying bias. Default is True.')
    
	parser.add_argument('--gcn_decay', type=float, default=0.01, help='Initial decay rate.')


	return parser.parse_args()


def evaluation(DATA, label, labeledRatio):
    
    UnlabeledRatio = 1 - labeledRatio
    print("test size is ", UnlabeledRatio)
    DATA_tr, DATA_te, tr_LABEL, te_LABEL = train_test_split(DATA, label, test_size=UnlabeledRatio, random_state=42)
    clf = KNeighborsClassifier(n_neighbors=10)
    clf.fit(DATA_tr, tr_LABEL)
    L_pred = clf.predict(DATA_te)
    # kmeans = KMeans(n_clusters=2, random_state=0).fit(DATA_tr)
    if te_LABEL.max() == 1:
        print("ACC: {:.3f}".format(accuracy_score(te_LABEL, L_pred, normalize=True)))
        print("F1-score: {:.3f}".format(f1_score(te_LABEL, L_pred)))
        print("AUC: {:.3f}".format(roc_auc_score(te_LABEL, L_pred)))
        print("ARI: {:.3f}".format(adjusted_rand_score(te_LABEL, L_pred)))
    else:
        print("ACC: {:.3f}".format(accuracy_score(te_LABEL, L_pred, normalize=True)))
        print("F1-weighted: {:.3f}".format(f1_score(te_LABEL, L_pred, average='weighted')))
        print("F1-macro: {:.3f}".format(f1_score(te_LABEL, L_pred, average='macro')))
        print("ARI: {:.3f}".format(adjusted_rand_score(te_LABEL, L_pred)))
    print(" ")



if __name__ == '__main__':
    args = parse_args()
    print(args)

    data_folder = args.data_folder
    device = torch.device('cuda')
    test_list = [0.8, 0.9, 0.95]

    print("\n ----Start to read data----")
    data = []
    omics_list = list(range(1, args.num_omics+1))
    data, labels = read_data(data_folder, omics_list)
    num_of_sample = len(labels)

    print("The input dataset is ", data_folder, " with omics number", len(omics_list))

    print("\n ----Start to build the graph----")

    topology= load_graph_data(data_folder, len(omics_list), num_of_sample, data)
    nx_G = read_graph(args, topology, num_of_sample)

    print(" ----Finish Building the graph---- \n ")

    DATA= train_RJ(args, num_of_sample, nx_G, topology, args.num_class, data_folder)

    evaluation(DATA, labels, args.labeled_ratio)	




    
    

    