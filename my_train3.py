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
from distutils.util import strtobool
import sys
sys.path.insert(0, '/home/kxie/cluster/DeProp')
from DeProp.model import DeProp
setup_seed(42)
torch.autograd.set_detect_anomaly(True)
print('start time:', datetime.now())


parser = argparse.ArgumentParser(description='my_model')
### training params ###
parser.add_argument('--dataset', type=str, default='cora', help="name of dataset")
parser.add_argument('--hidden_dim', type=int, default=64, help="hidden dimension")
parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
parser.add_argument('--H_lr', type=float, default=1e-3, help='Initial learning rate.')
parser.add_argument('--device', type=str, default='cuda:0', help='device')
parser.add_argument('--dropout', type=float, default=0.5, help='')


### encoding2 params ###
parser.add_argument('--first_transformation', type=str, default='mlp', help='mlp or linear')
parser.add_argument('--low_pass_layers', type=int, default=10, help='')
parser.add_argument('--alphaH', type=float, default=1.0, help='')
parser.add_argument('--alphaC', type=float, default=0.5, help='')
parser.add_argument('--alphaO', type=float, default=0.5, help='')

parser.add_argument('--high_pass_layers', type=int, default=5, help='')
parser.add_argument('--high_pass_alpha', type=float, default=1, help='')
parser.add_argument('--fusion_method', type=str, default='add', help='add, concat, max')
parser.add_argument('--fusion_beta', type=float, default=0.5, help='')
parser.add_argument('--fusion_gamma', type=float, default=1, help='')


### DeProp params ###
parser.add_argument('--gnnlayers', type=int, default=5, help="Number of gnn layers")
parser.add_argument('--step_size_gamma', type=float, default=0.005, help='')
parser.add_argument('--lambda1', type=float, default=100, help="")
parser.add_argument('--lambda2', type=float, default=0.03, help="")
parser.add_argument('--with_bn', type=strtobool, default=True, help="")
parser.add_argument('--F_norm', type=strtobool, default=True, help="")
parser.add_argument('--lin', type=strtobool, default=True, help="")


#### prop args ####
parser.add_argument('--Cprop', type=str, default='0', 
                    help='0: no propagation on C, \
                            lp: low pass, \
                            dep: deprop, \
                            dep_trans: deprop with transformation')
# parser.add_argument('--Hprop', type=str, default='lp', 
#                     help='lp: low pass,\
#                             dep: deprop, \
#                             dep_trans: deprop with transformation')



### loss params ###
parser.add_argument('--loss_lambda_prop', type=float, default=1, help='')
parser.add_argument('--sharpening', type=float, default=1, help="")
parser.add_argument('--loss_lambda_adj', type=float, default=1.0, help='')
parser.add_argument('--loss_lambda_attr', type=float, default=1.0, help='')
parser.add_argument('--loss_lambda_kmeans', type=float, default=0.1, help='')
# parser.add_argument('--reg_loss', type=str, default='orth', help='orth(ogonal), col(lapse), sqrt')
parser.add_argument('--kmeans_loss', type=str, default='tr', 
                    help='tr(ace), cen(troid contrastive), nod(e contrastive)')
parser.add_argument('--temperature', type=float, default=1, help='') 


### log params ###
parser.add_argument('--log_file', type=str, default='res_log/grid_search_deprop.txt', help='')

# parser.add_argument('--gamma', type=float, default=0.5, help='')
# parser.add_argument('--cluster_init_method', type=str, default='kmeans', help='random, kmeans, mlp')
# parser.add_argument('--pretrain_epochs', type=int, default=100, help='')
# parser.add_argument('--pretrain_lr', type=float, default=1e-3, help='')



args = parser.parse_args()

print(args)

##### load data #####
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
    ##### train #####
    best_acc = 0
    best_res = []
    cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)

    for e in range(args.epochs):
        # optimizer_C.zero_grad()
        optimizer_H.zero_grad()
        # if e > 50:
        #     scheduler.step()
        H = model(X, edge_index)

        # loss_prop = F.mse_loss(H @ H.T, X_prop @ X_prop.T)
        loss_prop = args.loss_lambda_prop * torch.pow(1 - cos_sim(H, X_prop), args.sharpening).mean()
        # loss_prop = F.mse_loss(H, X_prop)
        loss_adj = args.loss_lambda_adj * F.mse_loss(H @ H.T, org_adj)
        loss_attr = args.loss_lambda_attr * F.mse_loss(H @ H.T, X_norm @ X_norm.T)


        # evaluate the quality of the learned representation with kmeans
        # H = F.normalize(H, p=2, dim=1)
        cluster_ids,_ = k_means(H.detach().cpu(), cluster_num, device='cpu', distance='cosine')
        C0 = cluster_id2assignment(cluster_ids, cluster_num).to(args.device)
        C = Cprop(C0, A, args)
        C = F.softmax(C, dim=1) # use softmax or normalize?
        loss_kmeans = args.loss_lambda_kmeans * kmeans_loss_fn(C, H, args)
        
        loss = loss_prop + loss_kmeans
        loss.backward()
        optimizer_H.step()



        ## evaluation
        print('epoch: %d , loss: %.2f, loss_kmeans: %.2f, loss_prop: %.2f, loss_adj: %.2f, loss_attr: %.2f' % (e, loss.item(), loss_kmeans.item(), loss_prop, loss_adj.item(), loss_attr.item()), end=' ')
        predict_labels = torch.argmax(C, dim=1)
        res = print_eval(predict_labels.cpu().numpy(), true_labels.cpu().numpy(), A.cpu().numpy())

        ## best acc
        if res[0] > best_acc:
            best_acc = res[0]
            best_res = res
            best_e = e
    
    print(f'best epoch: {best_e}')
    return best_res



total_res = []
# final result is the mean of 5 repeated experiments
for fold in range(5):
    print("#"*60)
    print("#"*26, ' fold:%d ' % fold, "#"*26)
    print("#"*60)

    #### define model and optimizer #####
    model = DeProp(in_channels=X.shape[1], hidden_channels=args.hidden_dim, 
                   out_channels=args.hidden_dim, num_layers=args.gnnlayers, orth=True, 
                   lambda1=args.lambda1, lambda2=args.lambda2, gamma=args.step_size_gamma, 
                   with_bn=args.with_bn, F_norm=args.F_norm, dropout=args.dropout, lin=args.lin).to(args.device)
    # model = encoding2(args, X.shape[1], args.hidden_dim, A, args.hidden_dim).to(args.device)
    
    optimizer_H = torch.optim.Adam(model.parameters(), lr=args.H_lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_H, milestones=[50, 100, 150], gamma=0.5)


    ##### compute low rank X_prop #####
    X_norm = F.normalize(X, p=2, dim=1)
    X_prop = X_norm
    for _ in range(args.low_pass_layers):
        X_prop = args.alphaH * torch.spmm(A, X_prop) + X_norm
    U, S, _ = torch.svd_lowrank(X_prop, q=args.hidden_dim, niter=7)
    X_prop = U @ torch.diag(S)
    X_prop = F.normalize(X_prop, p=2, dim=1)

    ##### test the clustering quality on initial X_prop #####
    cluster_ids,_ = k_means(X_prop.detach(), cluster_num, device=args.device, distance='cosine')
    from utils import cluster_id2assignment, Cprop
    C0 = cluster_id2assignment(cluster_ids, cluster_num).to(args.device)
    C = Cprop(C0, A, args)
    # for _ in range(args.low_pass_layers):
    #     C = DePropagate(C, C0, A, args.step_size_gamma, args.alphaC, args.alphaO)
    #     C = F.normalize(C, p=2, dim=1)
    predict_labels = torch.argmax(C, dim=1)
    res = print_eval(predict_labels.cpu().numpy(), true_labels.cpu().numpy(), A.cpu().numpy())    


    ##### training and evaluation #####
    res = train()
    total_res.append(res)

    print('fold: %d, acc: %.2f, nmi: %.2f, ari: %.2f, f1: %.2f, mod: %.2f, con: %.2f' % (fold, res[0]*100, res[1]*100, res[2]*100, res[3]*100, res[4]*100, res[5]*100))

# save the result
with open(args.log_file, 'a+') as f:
    # if the file is empty, write the header
    if os.path.getsize(args.log_file) == 0:
        f.write('H_lr, loss_lambda_kmeans, gnnlayers, with_bn, F_norm, lin, step_size_gamma, lambda1, lambda2, dropout, ')
        f.write('dataset, acc_mean, acc_std, nmi_mean, nmi_std, ari_mean, ari_std, f1_mean, f1_std, mod_mean, mod_std, con_mean, con_std\n')
    f.write(', '.join([str(x) for x in [args.H_lr, args.loss_lambda_kmeans, args.gnnlayers, args.with_bn, args.F_norm, args.lin, args.step_size_gamma, args.lambda1, args.lambda2, args.dropout]])+', ')
    total_res = np.array(total_res)
    f.write('%s, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f\n'%(args.dataset, 
        total_res[:, 0].mean()*100, total_res[:, 0].std()*100, 
        total_res[:, 1].mean()*100, total_res[:, 1].std()*100, 
        total_res[:, 2].mean()*100, total_res[:, 2].std()*100, 
        total_res[:, 3].mean()*100, total_res[:, 3].std()*100, 
        total_res[:, 4].mean()*100, total_res[:, 4].std()*100, 
        total_res[:, 5].mean()*100, total_res[:, 5].std()*100))

print('***** final result: *****')
print('%s, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f\n'%(args.dataset, 
    total_res[:, 0].mean()*100, total_res[:, 0].std()*100, 
    total_res[:, 1].mean()*100, total_res[:, 1].std()*100, 
    total_res[:, 2].mean()*100, total_res[:, 2].std()*100, 
    total_res[:, 3].mean()*100, total_res[:, 3].std()*100, 
    total_res[:, 4].mean()*100, total_res[:, 4].std()*100, 
    total_res[:, 5].mean()*100, total_res[:, 5].std()*100))

print('end time:', datetime.now())
