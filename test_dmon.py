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
parser.add_argument('--device', type=str, default='cuda:0', help='device')

parser.add_argument('--first_transformation', type=str, default='mlp', help='mlp or linear')

parser.add_argument('--low_pass_layers', type=int, default=5, help='')
parser.add_argument('--low_pass_alpha', type=float, default=1, help='')
parser.add_argument('--high_pass_layers', type=int, default=5, help='')
parser.add_argument('--high_pass_alpha', type=float, default=1, help='')

parser.add_argument('--fusion_method', type=str, default='add', help='add, concat, max')
parser.add_argument('--fusion_beta', type=float, default=0.5, help='')
parser.add_argument('--fusion_gamma', type=float, default=1, help='')

parser.add_argument('--cluster_init_method', type=str, default='kmeans', help='random, kmeans, mlp')

parser.add_argument('--loss_lambda1', type=float, default=1, help='')
parser.add_argument('--loss_lambda2', type=float, default=1, help='')
parser.add_argument('--reg_loss', type=str, default='orth', help='orth(ogonal), col(lapse), sqrt')

parser.add_argument('--pretrain_epochs', type=int, default=100, help='')
parser.add_argument('--pretrain_lr', type=float, default=1e-3, help='')

parser.add_argument('--encoder', type=str, default='GCN', help='Using original GCN encoder (GCN) or our low-pass filter as encoder (low_pass)')

args = parser.parse_args()

X, y, A = load_graph_data(root_path='/home/kxie/cluster/dataset/', 
                          dataset_name=args.dataset, show_details=True)
cluster_num = len(np.unique(y))
edge_index = torch.LongTensor(np.array(A.nonzero())).to(args.device)

A = torch.FloatTensor(A).to(args.device)
A = normalize_adj_torch(A, self_loop=True, symmetry=True)
low_pass_filter = construct_filter(A, l=args.low_pass_layers, 
                                    alpha=args.low_pass_alpha)

X = torch.FloatTensor(X).to(args.device)
y = torch.LongTensor(y).to(args.device)
true_labels = y


def train():
    model = pool_based_model(args, X.shape[1], args.hidden_dim, cluster_num, 
                    low_pass_filter).to(args.device)
    optimizer_H = torch.optim.Adam(model.parameters(), lr=args.H_lr)

    ##### train #####
    best_acc = 0
    best_res = []
    for e in range(args.epochs):
        optimizer_H.zero_grad()

        C, out, out_adj, spectral_loss, ortho_loss, cluster_loss = model(X, A, edge_index)
        loss = spectral_loss + cluster_loss

        loss.backward()
        optimizer_H.step()

        ## evaluation
        print('epoch: %d, spec loss: %.4f, clus loss: %.4f' % (e, spectral_loss, cluster_loss), end=' ')
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
with open('/home/kxie/cluster/res_log/my_model.txt', 'a+') as f:
    total_res = np.array(total_res)
    f.write('%s, %.2f+-%.2f, %.2f+-%.2f, %.2f+-%.2f, %.2f+-%.2f, %.2f+-%.2f, %.2f+-%.2f\n'%(args.dataset, 
        total_res[:, 0].mean()*100, total_res[:, 0].std()*100, 
        total_res[:, 1].mean()*100, total_res[:, 1].std()*100, 
        total_res[:, 2].mean()*100, total_res[:, 2].std()*100, 
        total_res[:, 3].mean()*100, total_res[:, 3].std()*100, 
        total_res[:, 4].mean()*100, total_res[:, 4].std()*100, 
        total_res[:, 5].mean()*100, total_res[:, 5].std()*100))

print('end time:', datetime.now())