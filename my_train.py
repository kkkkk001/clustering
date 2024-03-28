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


args = parser.parse_args()

X, y, A = load_graph_data(root_path='/home/kxie/cluster/dataset/', 
                          dataset_name=args.dataset, show_details=True)
cluster_num = len(np.unique(y))

A = torch.FloatTensor(A).to(args.device)
edge_index = torch.LongTensor(np.array(A.nonzero())).to(args.device)
pdb.set_trace()
A = normalize_adj_torch(A, self_loop=True, symmetry=True)
low_pass_filter = construct_filter(A, l=args.low_pass_layers, 
                                    alpha=args.low_pass_alpha)

X = torch.FloatTensor(X).to(args.device)
y = torch.LongTensor(y).to(args.device)
true_labels = y


def train():
    model = encoding(args, X.shape[1], args.hidden_dim, 
                    low_pass_filter).to(args.device)
    optimizer_H = torch.optim.Adam(model.parameters(), lr=args.H_lr)

    ##### pretrained model #####
    if args.fusion_method == 'concat':
        # the out dimension of H is 2*hidden_dim because of concat op
        decoder = nn.Linear(args.hidden_dim*2, X.shape[1]).to(args.device)
    else:
        decoder = nn.Linear(args.hidden_dim, X.shape[1]).to(args.device)
    # wrap parameters from model and decoder
    optimizer_pretrain = torch.optim.Adam(list(model.parameters()) + 
                                        list(decoder.parameters()), 
                                        lr=args.pretrain_lr)
    for e in range(args.pretrain_epochs):
        optimizer_pretrain.zero_grad()
        H = model(X, A)
        # reconstruct adj matrix A with H
        loss1 = reconstruction_loss_distance(torch.mm(H, H.T), A)/A.shape[0]
        # reconstruct feature matrix X with decoder(H)
        loss2 = reconstruction_loss_cosine(decoder(H), X)/X.shape[0]
        loss = loss2
        loss.backward()
        optimizer_pretrain.step()
        print('PRETRAIN - epoch: %d, loss1: %.2f, loss2: %.2f' % (e, loss1.item(), loss2.item()), end=' ')
        
        # evaluate the quality of the learned representation with kmeans
        cluster_ids,_ = k_means(H.detach(), cluster_num, device='gpu', distance='cosine')
        print_eval(cluster_ids, true_labels.cpu().numpy(), A.cpu().numpy())
    ##############################



    H_init = model(X, A).detach()
    # H_init = model.fusion(model.low_pass_filter@X, model.high_pass_filter@X).detach()
    # H_init = A
    cluster = cluster_model(args.cluster_init_method, H_init.shape[0], args.hidden_dim, cluster_num, H_init).to(args.device)
    optimizer_C = torch.optim.Adam(cluster.parameters(), lr=args.C_lr)

    print('#'*20, "Initial Clustering", '#'*20)
    cluster_ids = torch.argmax(cluster(H_init), dim=1)
    initres = print_eval(cluster_ids.cpu().numpy(), true_labels.cpu().numpy(), A.cpu().numpy())
    print('#'*60)




    ##### train #####
    best_acc = 0
    best_res = []
    for e in range(args.epochs):
        optimizer_C.zero_grad()
        optimizer_H.zero_grad()

        H = model(X, A)
        C = cluster(H)
        # if args.cluster_init_method == 'mlp':
        loss1 = kmeans_loss_fn(H, C)
        loss2 = args.loss_lambda1 * spectral_loss_fn(C, A)
        loss3 = args.loss_lambda2 * reg_loss_fn(C, args)
        loss = loss1 + loss2 + loss3
        loss.backward()
        optimizer_H.step()
        optimizer_C.step()
        # else:
        #     ## fix C and update H
        #     loss2 = kmeans_loss_fn(H, C.detach())
        #     loss2.backward()
        #     optimizer_H.step()


        #     ## fix H and update C
        #     loss11 = kmeans_loss_fn(H.detach(), C)
        #     loss12 = args.loss_lambda1 * spectral_loss_fn(C, A)
        #     loss13 = args.loss_lambda2 * reg_loss_fn(C, args)
        #     loss1 = loss11 + loss12 + loss13
        #     loss1.backward()
        #     optimizer_C.step()


        ## evaluation
        print('epoch: %d' % e, end=' ')
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