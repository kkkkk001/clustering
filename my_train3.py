import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, pdb, sys, argparse
from dgc.utils import load_graph_data, normalize_adj, construct_filter, normalize_adj_torch
import scipy.sparse as sp
from model import *
from dgc.clustering import k_means
from dgc.eval import print_eval
from dgc.rand import setup_seed
from datetime import datetime
import sys
sys.path.insert(0, '/home/kxie/cluster/DeProp')
setup_seed(42)
torch.autograd.set_detect_anomaly(True)
print('start time:', datetime.now())


parser = argparse.ArgumentParser(description='my_model')
parser.add_argument('--dataset', type=str, default='cora', help="name of dataset")
parser.add_argument('--gnnlayers', type=int, default=3, help="Number of gnn layers")
parser.add_argument('--hidden_dim', type=int, default=64, help="hidden dimension")
parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train.')
parser.add_argument('--H_lr', type=float, default=1e-3, help='Initial learning rate.')
parser.add_argument('--C_lr', type=float, default=1e-3, help='Initial learning rate.')
parser.add_argument('--device', type=str, default='cpu:0', help='device')

parser.add_argument('--first_transformation', type=str, default='mlp', help='mlp or linear')

parser.add_argument('--low_pass_layers', type=int, default=10, help='')
parser.add_argument('--alphaH', type=float, default=1.0, help='')
parser.add_argument('--alphaC', type=float, default=0.5, help='')
parser.add_argument('--alphaO', type=float, default=0.5, help='')
parser.add_argument('--step_size_gamma', type=float, default=0.005, help='')

parser.add_argument('--high_pass_layers', type=int, default=5, help='')
parser.add_argument('--high_pass_alpha', type=float, default=1, help='')

parser.add_argument('--fusion_method', type=str, default='add', help='add, concat, max')
parser.add_argument('--fusion_beta', type=float, default=0.5, help='')
parser.add_argument('--fusion_gamma', type=float, default=1, help='')

parser.add_argument('--dropout', type=float, default=0.5, help='')
parser.add_argument('--gamma', type=float, default=0.5, help='')

parser.add_argument('--cluster_init_method', type=str, default='kmeans', help='random, kmeans, mlp')

parser.add_argument('--loss_lambda_adj', type=float, default=1.0, help='')
parser.add_argument('--loss_lambda_attr', type=float, default=1.0, help='')
parser.add_argument('--loss_lambda_kmeans', type=float, default=0.1, help='')

parser.add_argument('--reg_loss', type=str, default='orth', help='orth(ogonal), col(lapse), sqrt')
parser.add_argument('--kmeans_loss', type=str, default='tr', 
                    help='tr(ace), cen(troid contrastive), nod(e contrastive)')
parser.add_argument('--temperature', type=float, default=1, help='') 

parser.add_argument('--pretrain_epochs', type=int, default=100, help='')
parser.add_argument('--pretrain_lr', type=float, default=1e-3, help='')


#### test args ####
parser.add_argument('--Cprop', type=str, default='0', 
                    help='0: no propagation on C, \
                            lp: low pass, \
                            dep: deprop, \
                            dep_trans: deprop with transformation')
parser.add_argument('--Hprop', type=str, default='lp', 
                    help='lp: low pass,\
                            dep: deprop, \
                            dep_trans: deprop with transformation')



args = parser.parse_args()

X, y, A = load_graph_data(root_path='/home/kxie/cluster/dataset/', 
                          dataset_name=args.dataset, show_details=True)
cluster_num = len(np.unique(y))
edge_index = torch.LongTensor(np.array(A.nonzero())).to(args.device)

A = torch.FloatTensor(A).to(args.device)
org_adj = A
A = normalize_adj_torch(A, self_loop=True, symmetry=True)
low_pass_filter = construct_filter(A, l=args.low_pass_layers, 
                                    alpha=args.alphaH)

X = torch.FloatTensor(X).to(args.device)
y = torch.LongTensor(y).to(args.device)
true_labels = y


def train():
    from DeProp.model import DeProp
    DeProp_args = {}
    DeProp_args['num_layers'] = args.gnnlayers
    DeProp_args['hidden_dim'] = args.hidden_dim
    DeProp_args['orth'] = True
    DeProp_args['lambda1'] = args.alphaH
    DeProp_args['lambda2'] = args.alphaO
    DeProp_args['gamma'] = args.step_size_gamma
    DeProp_args['with_bn'] = True
    DeProp_args['F_norm'] = True
    DeProp_args['dropout'] = args.dropout
    DeProp_args['smooth'] = False

    DeProp_args = argparse.Namespace(**DeProp_args)
    model = DeProp(in_channels=X.shape[1], hidden_channels=args.hidden_dim, 
                   out_channels=args.hidden_dim, dropout=args.dropout, args=DeProp_args).to(args.device)

    model = encoding2(args, X.shape[1], args.hidden_dim, A, args.hidden_dim).to(args.device)
    optimizer_H = torch.optim.Adam(model.parameters(), lr=args.H_lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_H, milestones=[50, 100, 150], gamma=0.5)


    X_norm = F.normalize(X, p=2, dim=1)
    X_prop = X_norm
    for _ in range(args.low_pass_layers):
        X_prop = args.alphaH * torch.spmm(A, X_prop) + X_norm

    U, S, _ = torch.svd_lowrank(X_prop, q=args.hidden_dim, niter=7)
    X_prop = U @ torch.diag(S)
    X_prop = F.normalize(X_prop, p=2, dim=1)

    cluster_ids,_ = k_means(X_prop.detach(), cluster_num, device='cpu', distance='cosine')
    C = torch.zeros(X_prop.shape[0], cluster_num).to(args.device)
    C[torch.arange(X_prop.shape[0]), cluster_ids] = 1
    C = F.normalize(C, p=2, dim=0)
    C0 = C
    for _ in range(args.low_pass_layers):
        C = args.alphaC * torch.spmm(A, C) + C0

    # for _ in range(args.low_pass_layers):
    #     C = DePropagate(C, C0, A, args.step_size_gamma, args.alphaC, args.alphaO)
    #     C = F.normalize(C, p=2, dim=1)


    predict_labels = torch.argmax(C, dim=1)
    res = print_eval(predict_labels.cpu().numpy(), true_labels.cpu().numpy(), A.cpu().numpy())

    ##### train #####
    best_acc = 0
    best_res = []
    cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)

    for e in range(args.epochs):
        # optimizer_C.zero_grad()
        optimizer_H.zero_grad()
        if e > 50:
            scheduler.step()

        H = model(X, edge_index)

        # loss_prop = F.mse_loss(H @ H.T, X_prop @ X_prop.T)
        loss_prop = torch.pow(1 - cos_sim(H, X_prop), 1).mean()
        # loss_prop = F.mse_loss(H, X_prop)
        loss_adj = args.loss_lambda_adj * F.mse_loss(H @ H.T, org_adj)
        loss_attr = args.loss_lambda_attr * F.mse_loss(H @ H.T, X_norm @ X_norm.T)

        loss_kmeans = args.loss_lambda_kmeans * kmeans_loss_fn(C, H, args)
        
        # loss = loss_prop + loss_kmeans
        loss = loss_prop
        loss.backward()
        optimizer_H.step()

        # evaluate the quality of the learned representation with kmeans
        # H = F.normalize(H, p=2, dim=1)
        cluster_ids,_ = k_means(H.detach(), cluster_num, device='cpu', distance='cosine')
        # print_eval(cluster_ids, true_labels.cpu().numpy(), A.cpu().numpy())
        C = torch.zeros(H.shape[0], cluster_num).to(H.device)
        C[torch.arange(H.shape[0]), cluster_ids] = 1
        C = F.normalize(C, p=2, dim=0)
        C = F.softmax(C, dim=1)

        # C0 = C
        # for _ in range(args.low_pass_layers):
        #     C = args.alphaC * torch.spmm(A, C) + C0

        C = F.softmax(C, dim=1)

        ## evaluation
        print('epoch: %d , loss: %.2f, loss_kmeans: %.2f, loss_prop: %.2f, loss_adj: %.2f, loss_attr: %.2f' % (e, loss.item(), loss_kmeans.item(), loss_prop, loss_adj.item(), loss_attr.item()), end=' ')
        predict_labels = torch.argmax(C, dim=1)
        res = print_eval(predict_labels.cpu().numpy(), true_labels.cpu().numpy(), A.cpu().numpy())

        ## best acc
        if res[0] > best_acc:
            best_acc = res[0]
            best_res = res
    
    
    return best_res



total_res = []
# final result is the mean of 5 repeated experiments
for fold in range(5):
    print("#"*60)
    print("#"*26, ' fold:%d ' % fold, "#"*26)
    print("#"*60)

    res = train()
    total_res.append(res)

    print('fold: %d, acc: %.2f, nmi: %.2f, ari: %.2f, f1: %.2f, mod: %.2f, con: %.2f' % (fold, res[0]*100, res[1]*100, res[2]*100, res[3]*100, res[4]*100, res[5]*100))

# save the result
with open('./res_log/my_model.txt', 'a+') as f:
    total_res = np.array(total_res)
    f.write('%s, %.2f+-%.2f, %.2f+-%.2f, %.2f+-%.2f, %.2f+-%.2f, %.2f+-%.2f, %.2f+-%.2f\n'%(args.dataset, 
        total_res[:, 0].mean()*100, total_res[:, 0].std()*100, 
        total_res[:, 1].mean()*100, total_res[:, 1].std()*100, 
        total_res[:, 2].mean()*100, total_res[:, 2].std()*100, 
        total_res[:, 3].mean()*100, total_res[:, 3].std()*100, 
        total_res[:, 4].mean()*100, total_res[:, 4].std()*100, 
        total_res[:, 5].mean()*100, total_res[:, 5].std()*100))

print('end time:', datetime.now())
