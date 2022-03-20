from autoEncoder.TrainAE import *
from randomJump.randomJumpTrain import *
from graphConstruct.data_process import load_graph_data
import argparse

def parse_args():
	'''
	Parses the RandomJump arguments.
	'''
	parser = argparse.ArgumentParser(description="Run RandomJump.")

	# Parameters for the generation of walks
	parser.add_argument('--dimensions', type=int, default=64,
	                    help='Number of dimensions. Default is 64.')

	parser.add_argument('--walk-length', type=int, default=20,
	                    help='Length of walk per source. Default is 42.')

	parser.add_argument('--num-walks', type=int, default=10,
	                    help='Number of walks per source. Default is 10.')

	parser.add_argument('--window-size', type=int, default=10,
                    	help='Context size for optimization. Default is 10.')

	parser.add_argument('--iter', default=1, type=int,
                        help='Number of epochs in SGD')

	parser.add_argument('--workers', type=int, default=50,
	                    help='Number of parallel workers. Default is 8.')

	#Jumping probability hyperparamters
	parser.add_argument('--p', type=float, default=1,
	                    help='Return hyperparameter. Default is 1.')

	parser.add_argument('--q', type=float, default=2,
	                    help='Inout hyperparameter. Default is 2.')

	parser.add_argument('--z', type=float, default=0.7,
	                    help='Jump hyperparameter. Default is 0.7')

	parser.add_argument('--delta', type=float, default=0.1,
	                    help='Default Jump hyper-probablity. Default is 0.1.')

	#For the training of model for jump parameters 
	parser.add_argument('--Jump_epochs', type=int, default=1200,
	                    help='The number of epoch for jump parameter training')
	
	parser.add_argument('--Jump_lr', type=float, default=0.001,
	                    help='The learning rate for jump parameter training')

	parser.add_argument('--Jump_dropout', type=float, default=0.9,
	                    help='The dropout for jump parameter training')

	parser.add_argument('--Jump_bias', action='store_true', default=True,
	                    help='Boolean specifying bias. Default is True.')

	return parser.parse_args()


if __name__ == '__main__':
    data_folder = 'GSE156478_CITE'
    view_list = [1,2]
    num_epoch_pretrain = 500
    num_epoch = 2500
    lr_e_pretrain = 1e-3
    lr_e = 5e-4

    print("The input dataset is ", data_folder, " with view ", len(view_list))
    
    if data_folder == 'ROSMAP':
        num_class = 2
        adj_parameter = 2
        dim_he_list = [200,200,100]
    if data_folder == 'BRCA':
        num_class = 5
        adj_parameter = 10
        dim_he_list = [400,400,200]
    if data_folder == 'syn':
        num_class = 3
        adj_parameter = 10
        dim_he_list = [400,400,200]
    if data_folder == 'LGG':
        num_class = 2
        adj_parameter = 15
        dim_he_list = [1200,1200,400]
    if data_folder == 'GSE156478_CITE':
        num_class = 7
        adj_parameter = 18
        dim_he_list = [400,400,200]
    if data_folder == 'GSE156478_ASAP':
        num_class = 8
        adj_parameter = 18
        dim_he_list = [400,400,200]

    device = torch.device('cuda')
    args = parse_args()
	
    print("\n ----Start to train auto-encoders----\n ")

    embeddings, labels, trte_idx, sample_weight = train_test(data_folder, view_list, num_class,
                                                            lr_e_pretrain, lr_e, 
                                                            num_epoch_pretrain, num_epoch,
                                                            adj_parameter, dim_he_list)  

    print("\n ----Start to build the graph----")
    
    topology, topology_tr = load_graph_data(data_folder, len(view_list), labels, embeddings, device, trte_idx)

    print(" ----Finish Building the graph---- \n ")

    print("\n ----Start to implement random walk on graph----\n ")
    train_test_RJ(args, topology, topology_tr, len(view_list), trte_idx, labels, data_folder, sample_weight, num_class)

    
    

    