import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, pdb, sys, argparse
from dgc.utils import load_graph_data, normalize_adj, construct_filter, normalize_adj_torch
import scipy.sparse as sp
from model import *
from dgc.clustering import k_means
from dgc.eval import print_eval, match_cluster
from dgc.rand import setup_seed
from datetime import datetime
import time
from distutils.util import strtobool
import sys
sys.path.insert(0, '/home/kxie/cluster/DeProp')
from DeProp.model import DeProp
setup_seed(42)
torch.autograd.set_detect_anomaly(True)
start_time = time.time()
print('start time:', datetime.now())
from utils import cluster_id2assignment, Cprop
cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
from sklearn.metrics import accuracy_score, f1_score
from sklearn.cluster import KMeans
from torch.optim.lr_scheduler import CyclicLR
from torch_geometric.nn.models.mlp import MLP



def spectral_clu(H, cluster_num):
    from sklearn.cluster import SpectralClustering
    clustering = SpectralClustering(n_clusters=cluster_num, assign_labels='kmeans').fit(H)
    cluster_id = clustering.labels_
    cluster_centers = np.zeros((cluster_num, H.shape[1]))
    for i in range(cluster_num):
        cluster_centers[i] = H[cluster_id==i].mean(0)
    return cluster_id, cluster_centers



parser = argparse.ArgumentParser(description='my_model')
### training params ###
parser.add_argument('--dataset', type=str, default='cora', help="name of dataset")
parser.add_argument('--input_dim', type=int, default=64, help="hidden dimension")
parser.add_argument('--hidden_dim', type=int, default=64, help="hidden dimension")
parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
parser.add_argument('--t_lr', type=float, default=1e-3, help='Initial learning rate.')
parser.add_argument('--a_lr', type=float, default=1e-3, help='Initial learning rate.')
parser.add_argument('--t_wd', type=float, default=5e-4, help='')
parser.add_argument('--a_wd', type=float, default=5e-4, help='')
parser.add_argument('--device', type=str, default='cuda', help='device')
parser.add_argument('--dropout', type=float, default=0.0, help='')
parser.add_argument('--norm', type=int, default=0, help='')
parser.add_argument('--svd', type=int, default=1, help='if 1, svd on S_norm and A_norm, if 0, lin on S and A')
parser.add_argument('--svd_on_S', type=str, default='S_norm', help='obtain input features by svd on S or S_norm')
parser.add_argument('--svd_on_A', type=str, default='A_norm', help='obtain input features by svd on S or A_norm')
parser.add_argument('--post_process', type=str, default='l2-norm', help='if l2-norm, l2 normalization npn Ht and HA before fusion, if none, no post process')



### top_agg params ###
parser.add_argument('--top_layers', type=int, default=5, help='')
parser.add_argument('--top_alpha', type=float, default=0.5, help='')
parser.add_argument('--top_prop', type=str, default='sgc', help='sgc style, gcn style, appnp style, or deprop style')
# parser.add_argument('--top_linear_trans', type=int, default=1, help='1 for linear transformation, 0 for mlp transformation')
parser.add_argument('--top_linear_trans', type=str, default='lin', help='1 for linear transformation, 0 for mlp transformation')



### attr_agg params ###
parser.add_argument('--attr_layers', type=int, default=5, help='')
parser.add_argument('--attr_alpha', type=float, default=0.5, help='')
parser.add_argument('--attr_r', type=float, default=0.5, help='the nnz ratio of attr simi mtx')
parser.add_argument('--attr_prop', type=str, default='sgc', help='sgc style, gcn style, or appnp style')
# parser.add_argument('--attr_linear_trans', type=int, default=1, help='1 for linear transformation, 0 for mlp transformation')
parser.add_argument('--attr_linear_trans', type=str, default='lin', help='1 for linear transformation, 0 for mlp transformation')
parser.add_argument('--fusion_method', type=str, default='add', help='add, concat, max')
parser.add_argument('--fusion_beta', type=float, default=0.5, help='H = beta * H_t + (1-beta) * H_a')


### X_prop params ###
parser.add_argument('--xprop_layers', type=int, default=5, help='')
parser.add_argument('--xprop_alpha', type=float, default=0.5, help='')

### C_prop params ###
parser.add_argument('--cprop_layers', type=int, default=5, help='')
parser.add_argument('--cprop_alpha', type=float, default=0.2, help='')
parser.add_argument('--cprop_abl', type=int, default=0, help='')


### DeProp params ###
parser.add_argument('--gnnlayers', type=int, default=5, help="Number of gnn layers")
parser.add_argument('--step_size_gamma', type=float, default=0.005, help='')
parser.add_argument('--lambda1', type=float, default=100, help="")
parser.add_argument('--lambda2', type=float, default=0.03, help="")
parser.add_argument('--with_bn', type=strtobool, default=False, help="")
parser.add_argument('--F_norm', type=strtobool, default=True, help="")
parser.add_argument('--lin', type=strtobool, default=True, help="")


### loss params ###
parser.add_argument('--loss_lambda_prop', type=float, default=1, help='')
parser.add_argument('--sharpening', type=float, default=1, help="")
parser.add_argument('--loss_lambda_adj', type=float, default=1.0, help='')
parser.add_argument('--loss_lambda_attr', type=float, default=1.0, help='')
parser.add_argument('--loss_lambda_kmeans', type=float, default=0.1, help='')
# parser.add_argument('--reg_loss', type=str, default='orth', help='orth(ogonal), col(lapse), sqrt')
parser.add_argument('--kmeans_loss', type=str, default='tr', 
                    help='tr(ace), cen(troid contrastive), nod(e contrastive)')
parser.add_argument('--loss_lambda_SSG0', type=float, default=0.001, help='')
parser.add_argument('--loss_lambda_SSG1', type=float, default=0.01, help='')
parser.add_argument('--loss_lambda_SSG2', type=float, default=0.01, help='')
parser.add_argument('--loss_lambda_SSG3', type=float, default=0.01, help='')
parser.add_argument('--temperature', type=float, default=1, help='') 
parser.add_argument('--clu_size', type=strtobool, default=True, help='') 
parser.add_argument('--rounding', type=strtobool, default=False, help='') 


### log params ###
parser.add_argument('--log_file', type=str, default=None, help='')
parser.add_argument('--log_fold_file', type=str, default=None, help='')
parser.add_argument('--log_title', type=strtobool, default=True, help='')
parser.add_argument('--fold', type=str, default='0-1-2-3-4', help='num of repeats, and seed of each repeat, separated by -')


args = parser.parse_args()
if args.log_file is None:
    args.log_file = 'res_log/'+args.dataset+'_grid_search.txt'
if args.log_fold_file is None:
    args.log_fold_file = 'res_fold_log/'+args.dataset+'_grid_search.txt'

print(args)
args_keys = []
args_values = []
for key, value in vars(args).items():
    args_keys.append(key)
    args_values.append(str(value))

##### load data #####
X, true_labels, A = load_graph_data(root_path='/home/kxie/cluster/dataset/', 
                          dataset_name=args.dataset, show_details=True)
cluster_num = len(np.unique(true_labels))


edge_index = torch.LongTensor(np.array(A.nonzero())).to(args.device)
A = torch.FloatTensor(A).to(args.device)
A_norm = normalize_adj_torch(A, self_loop=True, symmetry=True)

X = torch.FloatTensor(X).to(args.device)
X_norm = F.normalize(X, p=2, dim=1)

S = compute_attr_simi_mtx(X, args.attr_r).to(args.device)
S_norm = normalize_adj_torch(S, self_loop=False, symmetry=True).to(args.device)


def train():
    ##### train #####
    best_acc = 0
    best_res = []

    beta = args.fusion_beta
    for e in range(args.epochs):
        optimizer_t.zero_grad()
        optimizer_a.zero_grad()

        pdb.set_trace()

        if args.svd == 1:
            global US_norm, UA_norm
        if args.svd == 0:
            US_norm = model.agg(lint(US_norm_))
            UA_norm = attr_model.agg(lina(UA_norm_))


        H_t = model(US_norm)
        H_a = attr_model(UA_norm)


        if args.post_process == 'l2-norm':
            print('l2-norm')
            H_t = F.normalize(H_t, p=2, dim=1)
            H_a = F.normalize(H_a, p=2, dim=1)

        H = fusion_attr(H_t, H_a) 
        # check if H contains nan
        if torch.isnan(H).sum() > 0:
            print('nan in H')
            pdb.set_trace()


        loss_prop = torch.pow(1 - cos_sim(H, X_prop), args.sharpening).mean()

        if args.rounding:
            print('rounding')
            if e == 0:
                cluster_ids,centers_xprop = k_means(X_prop.detach().cpu(), cluster_num, device='cpu', distance='cosine')
                cluster_ids,centers = k_means((torch.round(torch.tanh(H)*7)).detach().cpu(), cluster_num, device='cpu', distance='cosine', centers=centers_xprop)
            else:
                cluster_ids,centers = k_means((torch.round(torch.tanh(H)*7)).detach().cpu(), cluster_num, device='cpu', distance='cosine', centers='kmeans')
        else:
            if e == 0:
                cluster_ids,centers_xprop = k_means(X_prop.detach().cpu(), cluster_num, device='cpu', distance='cosine')
                cluster_ids,centers = k_means(H.detach().cpu(), cluster_num, device='cpu', distance='cosine', centers=centers_xprop)
            else:
                cluster_ids,centers = k_means(H.detach().cpu(), cluster_num, device='cpu', distance='cosine', centers='kmeans')




        C0 = cluster_id2assignment(cluster_ids, cluster_num).to(args.device)
        # C = Cprop(C0, A, args)
        if args.cprop_abl == 1:
            print('cprop_abl')
            C = C0
        else:
            C = C_prop_model(C0)
        # C = C0
        C = F.normalize(C, p=2, dim=1)


        loss_kmeans =  kmeans_loss_fn(H, C, args)
        


        if args.norm == 1:
            H_t_norm = (H_t - H_t.mean(dim=0)) / H_t.std(dim=0) / torch.sqrt(torch.tensor(H_t.shape[1]).to(H_t.device))
            H_a_norm = (H_a - H_a.mean(dim=0)) / H_a.std(dim=0) / torch.sqrt(torch.tensor(H_a.shape[1]).to(H_a.device))
        elif args.norm == 0:
            H_t_norm = H_t
            H_a_norm = H_a
        pdb.set_trace()
        ort_loss = (ortho_loss_fn(H_t_norm) + ortho_loss_fn(H_a_norm))
        # inv_loss_o = args.loss_lambda_SSG1 * (H_t_norm - H_a_norm).pow(2).sum()
        inv_loss_o = F.mse_loss(H_t_norm, H_a_norm)
        # inv_loss_o = args.loss_lambda_SSG1 * (H_t_norm - H_a_norm).norm(p=2, dim=1).mean()
        inv_loss_n = (node_t_neighbor_a_loss_fn2(H_t_norm, H_a_norm, A_ori=A) + node_t_neighbor_a_loss_fn2(H_a_norm, H_t_norm, A_ori=A))
        # print(C.sum(0).mean(), C.sum(0).max())
        inv_loss_c = (node_t_cluster_a_loss_fn2(H_t_norm, H_a_norm, C, clu_size=args.clu_size) + node_t_cluster_a_loss_fn2(H_a_norm, H_t_norm, C, clu_size=args.clu_size))
        inv_loss = args.loss_lambda_SSG1 * inv_loss_o + args.loss_lambda_SSG2 * inv_loss_n + args.loss_lambda_SSG3 * inv_loss_c


        loss = args.loss_lambda_prop * loss_prop + args.loss_lambda_kmeans * loss_kmeans + args.loss_lambda_SSG0 * ort_loss + inv_loss
        loss.backward()
        optimizer_t.step()
        optimizer_a.step()


        ## evaluation
        print('epoch: %d , loss: %.3f, loss_kmeans: %.3f, loss_prop: %.3f, ort_loss: %.3f,' % (e, loss.item(), loss_kmeans.item(), loss_prop, ort_loss), end=' ')
        print('inv_loss1: %.3f, inv_loss2: %.3f, inv_loss3: %.3f,' % (inv_loss_o, inv_loss_n, inv_loss_c), end=' ')
        predict_labels = torch.argmax(C, dim=1)
        res = print_eval(predict_labels.cpu().numpy(), true_labels, A.cpu().numpy())
        

        ## best acc
        if res[0] > best_acc:
            best_acc = res[0]
            best_res = res
            best_e = e
    print(f'best epoch: {best_e}')
    return best_res




retain_grah=True 
total_res = []
# for fold in range(5):
# final result is the mean of 5 repeated experiments
for fold in [int(x) for x in args.fold.split('-')]:
    setup_seed(43+fold)
    print("#"*60)
    print("#"*26, ' fold:%d ' % fold, "#"*26)
    print("#"*60)


    #### define model and optimizer #####
    model = top_agg(A_norm, args.top_alpha, args.top_layers, args.input_dim, args.hidden_dim, linear_prop=args.top_prop, linear_trans=args.top_linear_trans).to(args.device)
    attr_model = attr_agg(S_norm, args.attr_alpha, args.attr_layers, args.input_dim, args.hidden_dim, linear_prop=args.attr_prop, linear_trans=args.attr_linear_trans).to(args.device)

    fusion_attr = fusion(args.fusion_method, args.fusion_beta, args.hidden_dim).to(args.device)
    
    C_prop_model = C_agg(args.cprop_alpha, args.cprop_layers, A_norm).to(args.device)


    # lint = nn.Linear(X.shape[0], args.input_dim).to(args.device)
    # lina = nn.Linear(X.shape[0], args.input_dim).to(args.device)

    lint = MLP(in_channels=X.shape[0], hidden_channels=512, out_channels=args.hidden_dim, num_layers=2, batch_norm=False, dropout=0.0, bias=True).to(args.device)
    lina = MLP(in_channels=X.shape[0], hidden_channels=512, out_channels=args.hidden_dim, num_layers=2, batch_norm=False, dropout=0.0, bias=True).to(args.device)
    

    optimizer_t = torch.optim.Adam(list(model.parameters())+list(lint.parameters()), lr=args.t_lr, weight_decay=args.t_wd)
    optimizer_a = torch.optim.Adam(list(attr_model.parameters())+list(lina.parameters()), lr=args.a_lr, weight_decay=args.a_wd)


    ##### compute low rank X_prop #####
    X_prop = X_norm
    for _ in range(args.xprop_layers):
        X_prop = args.xprop_alpha * torch.spmm(A_norm, X_prop) + X_norm
    U, s, _ = torch.svd_lowrank(X_prop, q=args.hidden_dim, niter=7)
    X_prop = U @ torch.diag(s)
    X_prop = F.normalize(X_prop, p=2, dim=1)
    pdb.set_trace()

    ##### compute low rank US_norm and UA_norm #####
    if args.svd == 1:
        US_norm, _, _ = torch.svd_lowrank(eval(args.svd_on_S), q=args.input_dim, niter=7)
        UA_norm, _, _ = torch.svd_lowrank(eval(args.svd_on_A), q=args.input_dim, niter=7)
        US_norm = model.agg(US_norm)
        UA_norm = attr_model.agg(UA_norm)
    elif args.svd == 0:
        US_norm_ = X@X.t()
        UA_norm_ = A 
        US_norm = model.agg(lint(US_norm_))
        UA_norm = attr_model.agg(lina(UA_norm_))

    # pdb.set_trace()


    ##### test init clustering quality #####
    ##### on X #####
    print('clustering results on initial X:\t', end=' ')
    cluster_ids,_ = k_means(X, cluster_num, device='cpu', distance='cosine')
    print_eval(cluster_ids, true_labels, A.cpu().numpy())   

    ##### on X_prop #####
    print('clustering results on initial X_prop:\t', end=' ')
    cluster_ids,centers = k_means(X_prop.detach(), cluster_num, device='cpu', distance='cosine')
    print_eval(cluster_ids, true_labels, A.cpu().numpy())  

    ##### on H_t #####
    print('clustering results on initial H_t:\t', end=' ')
    cluster_ids,centers = k_means((model.top_filter @ US_norm).detach(), cluster_num, device='cpu', distance='cosine')
    print_eval(cluster_ids, true_labels, A.cpu().numpy()) 

    ##### on H_a #####
    print('clustering results on initial H_a:\t', end=' ')
    cluster_ids,centers = k_means((attr_model.attr_filter @ UA_norm).detach(), cluster_num, device='cpu', distance='cosine')
    print_eval(cluster_ids, true_labels, A.cpu().numpy())  


    ##### training and evaluation #####
    res = train()
    total_res.append(res)

    print('fold: %d, acc: %.2f, nmi: %.2f, ari: %.2f, f1: %.2f, mod: %.2f, con: %.2f' % (fold, res[0]*100, res[1]*100, res[2]*100, res[3]*100, res[4]*100, res[5]*100))


total_res = np.array(total_res)

print('***** final result: *****')
print('%s, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f\n'%(args.dataset, 
    total_res[:, 0].mean()*100, total_res[:, 0].std()*100, 
    total_res[:, 1].mean()*100, total_res[:, 1].std()*100, 
    total_res[:, 2].mean()*100, total_res[:, 2].std()*100, 
    total_res[:, 3].mean()*100, total_res[:, 3].std()*100, 
    total_res[:, 4].mean()*100, total_res[:, 4].std()*100, 
    total_res[:, 5].mean()*100, total_res[:, 5].std()*100))

print(total_res)


total_time = time.time() - start_time
print('total time:', total_time)


# save the result
with open(args.log_file, 'a+') as f:
    # if the file is empty, write the header
    if args.log_title:
        f.write('norm, layers, attr_layers, alpha, S_alpha, ssg0, ssg1, ssg2, ssg3, fusion_beta, cprop_abl, svd, post_process, ')
        f.write('dataset, acc_mean, acc_std, nmi_mean, nmi_std, ari_mean, ari_std, f1_mean, f1_std, mod_mean, mod_std, con_mean, con_std, time, ')
        f.write(', '.join(x for x in args_keys)+'\n')
    f.write(', '.join([str(x) for x in [args.norm, args.top_layers, args.attr_layers, args.top_alpha, args.attr_alpha, args.loss_lambda_SSG0, args.loss_lambda_SSG1, args.loss_lambda_SSG2, args.loss_lambda_SSG3, args.fusion_beta, args.cprop_abl, args.svd]])+', '+args.post_process+', ')
    f.write('%s, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, '%(args.dataset, 
        total_res[:, 0].mean()*100, total_res[:, 0].std()*100, 
        total_res[:, 1].mean()*100, total_res[:, 1].std()*100, 
        total_res[:, 2].mean()*100, total_res[:, 2].std()*100, 
        total_res[:, 3].mean()*100, total_res[:, 3].std()*100, 
        total_res[:, 4].mean()*100, total_res[:, 4].std()*100, 
        total_res[:, 5].mean()*100, total_res[:, 5].std()*100, total_time))
    f.write(', '.join(x for x in args_values)+'\n')
    

# save the result of each fold
with open(args.log_fold_file, 'a+') as f:
    # if the file is empty, write the header
    if args.log_title:
        f.write('norm, layers, attr_layers, alpha, S_alpha, ssg0, ssg1, ssg2, ssg3, fusion_beta, cprop_abl, svd, post_process, ')
        f.write('dataset, fold, acc_mean, nmi_mean, ari_mean, f1_mean, mod_mean, con_mean, time, ')
        f.write(', '.join(x for x in args_keys)+'\n')
    for i in range(total_res.shape[0]):
        f.write(', '.join([str(x) for x in [args.norm, args.top_layers, args.attr_layers, args.top_alpha, args.attr_alpha, args.loss_lambda_SSG0, args.loss_lambda_SSG1, args.loss_lambda_SSG2, args.loss_lambda_SSG3, args.fusion_beta, args.cprop_abl, args.svd]])+', '+args.post_process+', ')
        f.write('%s, %d, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, '%(args.dataset, i, 
            total_res[i, 0]*100, total_res[i, 1]*100, total_res[i, 2]*100, total_res[i, 3]*100, total_res[i, 4]*100, total_res[i, 5]*100, total_time))
        f.write(', '.join(x for x in args_values)+'\n')



print('end time:', datetime.now())


