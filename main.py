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

	parser.add_argument('--walk-length', type=int, default=42,
	                    help='Length of walk per source. Default is 42.')

	parser.add_argument('--num-walks', type=int, default=20,
	                    help='Number of walks per source. Default is 20.')

	parser.add_argument('--window-size', type=int, default=10,
                    	help='Context size for optimization. Default is 10.')

	parser.add_argument('--iter', default=1, type=int,
                        help='Number of epochs in SGD')

	parser.add_argument('--workers', type=int, default=100,
	                    help='Number of parallel workers. Default is 100.')

	#Jumping probability hyperparamters
	parser.add_argument('--p', type=float, default=1,
	                    help='Return hyperparameter. Default is 1.')

	parser.add_argument('--q', type=float, default=2,
	                    help='Inout hyperparameter. Default is 2.')

	parser.add_argument('--z', type=float, default=0.7,
	                    help='Jump hyperparameter. Default is 0.7.')

	parser.add_argument('--delta', type=float, default=0.1,
	                    help='Default Jump hyper-probablity. Default is 0.1.')

	#For the training of model for jump parameters 
	parser.add_argument('--Jump_epochs', type=int, default=20,
	                    help='The number of epoch for jump parameter training')
	
	parser.add_argument('--Jump_lr', type=float, default=0.0001,
	                    help='The learning rate for jump parameter training')

	return parser.parse_args()


def prepare_trte_data(data_folder, view_list):
    data_list = []
    labels = np.loadtxt(os.path.join(data_folder, "labels.csv"), delimiter=',')
    for i in view_list:
        data = np.loadtxt(os.path.join(data_folder, str(i)+".csv"), delimiter=',')
        print("Omics", i, "is of shape", data.shape)
        data_list.append(data)
    
    return data_list, labels

if __name__ == '__main__':
    data_folder = ' '

    view_list = [1,2]
    num_class = 0

    print("The input dataset is ", data_folder, " with view ", len(view_list))

    device = torch.device('cuda')
    args = parse_args()
    test_list = [0.9]
    print(args)
    print("\n ----Start to prepare data----")
    num_view = len(view_list)
    data, labels = prepare_trte_data(data_folder, view_list)
    num_of_nodes = len(labels)
    print("\n ----Start to build the graph----")
    
    topology= load_graph_data(data_folder, len(view_list), num_of_nodes, data)

    print(" ----Finish Building the graph---- \n ")

    print("\n ----Start to implement random walk on graph----\n ")
    RJ(args, topology, len(view_list), labels, test_list, data_folder, num_class)

    
    

    