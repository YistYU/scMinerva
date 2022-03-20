import torch 
import torch.nn as nn

def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
           m.bias.data.fill_(0.0)


class CNN(torch.nn.Module):

    def __init__(self, args):
        super(CNN, self).__init__()
        self.dim = args.nn_dim
        self.fc1 = torch.nn.Linear(self.dim, self.dim, bias=args.Jump_bias)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.layer4 = torch.nn.Sequential(
            self.fc1,
            torch.nn.Sigmoid(),
            torch.nn.Dropout(p=args.Jump_dropout))

    def forward(self, args, x):
        num_sample = x.shape[0]
        num_fold = x.shape[1]
        self.dim = x.shape[1]
        out = self.layer4(x)
        out = out.reshape(num_sample, num_fold)
        return out


def initialize_train_para_vec(args, num_class):
    model = CNN(args)
    learning_rate = args.Jump_lr
    criterion = torch.nn.MSELoss()  
    optim = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    return model, criterion, optim

def train_para_vec(args, para_vec, model, cost, optimizer):
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
