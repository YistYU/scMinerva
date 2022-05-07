import torch 
import torch.nn as nn

import numpy as np
from torch import Tensor
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid



class GCN(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.in_channel = (int) (args.dimensions)
        self.latent_layer = (int) (self.in_channel / 4)
        self.dim = args.nn_dim
        self.conv1 = GCNConv(self.in_channel, self.latent_layer)
        self.conv2 = GCNConv(self.latent_layer, self.dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index).relu()
        return x

def initialize_train_para_vec(args, num_class):
    model = GCN(args)
    learning_rate = args.Jump_lr
    optim = torch.optim.SGD(params=model.parameters(), lr=learning_rate, momentum=0.1)
    return model, optim

def train_para_vec(args, para_vec, model, cost, optimizer):
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
