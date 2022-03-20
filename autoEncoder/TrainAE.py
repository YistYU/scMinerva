""" Training and testing of the model
"""
import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torch
import torch.nn.functional as F
from autoEncoder.ae_training_models import init_model_dict, init_optim
from autoEncoder.utils import one_hot_tensor, cal_sample_weight, gen_adj_mat_tensor, gen_test_adj_mat_tensor, cal_adj_mat_parameter
from randomJump.evaluate import *
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from tqdm._tqdm import trange

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
cuda = True if torch.cuda.is_available() else False


# def prepare_trte_data(data_folder, view_list):
#     num_view = len(view_list)
#     #With Redundant
#     #labels = np.loadtxt(os.path.join(data_folder, "labels.csv"), delimiter=',', usecols=(1))
#     #Without Redundant
#     labels = np.loadtxt(os.path.join(data_folder, "labels.csv"), delimiter=',')
    
#     data_tr_list = []
#     data_te_list = []
#     data_trte_list = []
#     for i in view_list:
#         # Without redundant
#         # data = np.loadtxt(os.path.join(data_folder, str(i)+".csv"))
#         # data_trte_list.append(data)
#         # With redundant
#         with open(os.path.join(data_folder, str(i)+".csv")) as f:
#             ncols = len(f.readline().split(','))
#         data = np.loadtxt(os.path.join(data_folder, str(i)+".csv"), delimiter=',', usecols=range(1, ncols), skiprows=1)
#         #data = data.T
#         print("Omics", i, "is of shape", data.shape)
#         data_trte_list.append(data)

#         # Finish removing the redundant
#         labels_tr, labels_te, data_tr, data_te = train_test_split(labels, data_trte_list[i-1], test_size=0.3, random_state=4)
#         data_tr_list.append(data_tr)
#         data_te_list.append(data_te)
#     labels_tr = labels_tr.astype(int)
#     labels_te = labels_te.astype(int)
#     num_tr = data_tr_list[0].shape[0]
#     num_te = data_te_list[0].shape[0]
#     data_mat_list = []
#     for i in range(num_view):
#         data_mat_list.append(np.concatenate((data_tr_list[i], data_te_list[i]), axis=0))
#     data_tensor_list = []
#     for i in range(len(data_mat_list)):
#         data_tensor_list.append(torch.FloatTensor(data_mat_list[i]))
#         if cuda:
#             data_tensor_list[i] = data_tensor_list[i].cuda()
#     idx_dict = {}
#     idx_dict["tr"] = list(range(num_tr))
#     idx_dict["te"] = list(range(num_tr, (num_tr+num_te)))
#     data_train_list = []
#     data_all_list = []
#     for i in range(len(data_tensor_list)):
#         data_train_list.append(data_tensor_list[i][idx_dict["tr"]].clone())
#         data_all_list.append(torch.cat((data_tensor_list[i][idx_dict["tr"]].clone(),
#                                        data_tensor_list[i][idx_dict["te"]].clone()),0))
#     labels = np.concatenate((labels_tr, labels_te))
    
#     return data_train_list, data_all_list, idx_dict, labels



def prepare_trte_data(data_folder, view_list):
    num_view = len(view_list)
    labels_tr = np.loadtxt(os.path.join(data_folder, "labels_tr.csv"), delimiter=',')
    labels_te = np.loadtxt(os.path.join(data_folder, "labels_te.csv"), delimiter=',')
    labels_tr = labels_tr.astype(int)
    labels_te = labels_te.astype(int)
    data_tr_list = []
    data_te_list = []
    for i in view_list:
        data_tr_list.append(np.loadtxt(os.path.join(data_folder, str(i)+"_tr.csv"), delimiter=','))
        data_te_list.append(np.loadtxt(os.path.join(data_folder, str(i)+"_te.csv"), delimiter=','))
    num_tr = data_tr_list[0].shape[0]
    num_te = data_te_list[0].shape[0]
    data_mat_list = []
    for i in range(num_view):
        data_mat_list.append(np.concatenate((data_tr_list[i], data_te_list[i]), axis=0))
    data_tensor_list = []
    for i in range(len(data_mat_list)):
        data_tensor_list.append(torch.FloatTensor(data_mat_list[i]))
        if cuda:
            data_tensor_list[i] = data_tensor_list[i].cuda()
    idx_dict = {}
    idx_dict["tr"] = list(range(num_tr))
    idx_dict["te"] = list(range(num_tr, (num_tr+num_te)))
    data_train_list = []
    data_all_list = []
    for i in range(len(data_tensor_list)):
        data_train_list.append(data_tensor_list[i][idx_dict["tr"]].clone())
        data_all_list.append(torch.cat((data_tensor_list[i][idx_dict["tr"]].clone(),
                                       data_tensor_list[i][idx_dict["te"]].clone()),0))
    labels = np.concatenate((labels_tr, labels_te))
    
    return data_train_list, data_all_list, idx_dict, labels

def test_epoch(data_list, adj_list, te_idx, model_dict):
    for m in model_dict:
        model_dict[m].eval()
    num_view = len(data_list)
    ci_list = []
    for i in range(num_view):
        ci_list.append(model_dict["C{:}".format(i+1)](model_dict["E{:}".format(i+1)](data_list[i],adj_list[i])))
    if num_view >= 2:
        c = model_dict["C"](ci_list)    
    else:
        c = ci_list[0]
    c = c[te_idx,:]
    prob = F.softmax(c, dim=1).data.cpu().numpy()
    
    return prob


def gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx, adj_parameter):
    adj_metric = "cosine" # cosine distance
    adj_train_list = []
    adj_test_list = []
    adj_trte_list = []
    for i in range(len(data_tr_list)):
        adj_parameter_adaptive = cal_adj_mat_parameter(adj_parameter, data_tr_list[i], adj_metric)
        adj_train_list.append(gen_adj_mat_tensor(data_tr_list[i], adj_parameter_adaptive, adj_metric))
        adj_test_list.append(gen_test_adj_mat_tensor(data_trte_list[i], trte_idx, adj_parameter_adaptive, adj_metric))
        adj_trte_list.append(gen_adj_mat_tensor(data_trte_list[i], adj_parameter_adaptive, adj_metric))
    
    return adj_train_list, adj_test_list, adj_trte_list


def train_epoch(data_list, adj_list, label, one_hot_label, sample_weight, model_dict, optim_dict):
    loss_dict = {}
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    for m in model_dict:
        model_dict[m].train()    
    num_view = len(data_list)
    for i in range(num_view):
        optim_dict["C{:}".format(i+1)].zero_grad()
        ci_loss = 0
        ci = model_dict["C{:}".format(i+1)](model_dict["E{:}".format(i+1)](data_list[i],adj_list[i]))
        ci_loss = torch.mean(torch.mul(criterion(ci, label),sample_weight))
        ci_loss.backward()
        optim_dict["C{:}".format(i+1)].step()
        loss_dict["C{:}".format(i+1)] = ci_loss.detach().cpu().numpy().item()
    
    return loss_dict
 


def train_test(data_folder, view_list, num_class,
               lr_e_pretrain, lr_e, 
               num_epoch_pretrain, num_epoch,
               adj_parameter, dim_he_list,
               ):
    test_inverval = 50
    num_view = len(view_list)
    dim_hvcdn = pow(num_class,num_view)
    data_tr_list, data_trte_list, trte_idx, labels_trte = prepare_trte_data(data_folder, view_list)
    
    labels_tr_tensor = torch.LongTensor(labels_trte[trte_idx["tr"]])
    onehot_labels_tr_tensor = one_hot_tensor(labels_tr_tensor, num_class)
    sample_weight_tr = cal_sample_weight(labels_trte[trte_idx["tr"]], num_class)
    sample_weight_tr = torch.FloatTensor(sample_weight_tr)
    if cuda:
        labels_tr_tensor = labels_tr_tensor.cuda()
        onehot_labels_tr_tensor = onehot_labels_tr_tensor.cuda()
        sample_weight_tr = sample_weight_tr.cuda()
    adj_tr_list, adj_te_list, adj_trte_list = gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx, adj_parameter)
    dim_list = [x.shape[1] for x in data_tr_list]
    model_dict = init_model_dict(num_view, num_class, dim_list, dim_he_list)

    filename = data_folder + "_tr" + ".csv"
    tr_label = np.array(trte_idx["tr"])
    np.savetxt(os.path.join(data_folder,filename), tr_label, delimiter=",")
    filename = data_folder + "_te" + ".csv"
    te_label = np.array(trte_idx["te"])
    np.savetxt(os.path.join(data_folder,filename), te_label, delimiter=",")

    for m in model_dict:
        if cuda:
            model_dict[m].cuda()
    
    print("Pretrain GCNs...")
    optim_dict = init_optim(num_view, model_dict, lr_e_pretrain)
    for epoch in range(num_epoch_pretrain):
        train_epoch(data_tr_list, adj_tr_list, labels_tr_tensor, 
                    onehot_labels_tr_tensor, sample_weight_tr, model_dict, optim_dict)
    
    print("Finish pretraining, start training...")
    optim_dict = init_optim(num_view, model_dict, lr_e)
    for epoch in trange(num_epoch+1):
        train_epoch(data_tr_list, adj_tr_list, labels_tr_tensor, 
                    onehot_labels_tr_tensor, sample_weight_tr, model_dict, optim_dict)
    print("Data for training is", len(trte_idx["tr"]), ", data for tesing is", len(trte_idx["te"]))

    emb_list = []
    for i in range(num_view):
        embeddings = model_dict["C{:}".format(i+1)](model_dict["E{:}".format(i+1)](data_trte_list[i], adj_trte_list[i]))
        embeddings = embeddings.cpu().detach().numpy()
        emb_list.append(embeddings)
        filename = data_folder + "_" + str(i) + ".csv"

        np.savetxt(os.path.join(data_folder,filename), embeddings, delimiter=",")
    
    filename = data_folder + "_label"  + ".csv"
    labels = np.array(labels_trte)
    np.savetxt(os.path.join(data_folder,filename), labels, delimiter=",")

    return emb_list, labels, trte_idx, sample_weight_tr

    