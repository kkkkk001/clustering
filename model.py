from dgc.utils import load_graph_data, normalize_adj, construct_filter, normalize_adj_torch
from dgc.clustering import k_means
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
from torch_geometric.nn import DMoNPooling, GCNConv

from torch_geometric.nn.models.mlp import MLP



class encoding(nn.Module):
    def __init__(self, args, input_dim, hidden_dim, low_pass_filter):
        super(encoding, self).__init__()
        self.args = args

        if args.first_transformation=='mlp':
            self.first_layer = nn.Sequential(nn.Linear(input_dim, hidden_dim), 
                                             nn.ReLU(),
                                             nn.Linear(hidden_dim, hidden_dim))
        elif args.first_transformation=='linear':
            self.first_layer = nn.Linear(input_dim, hidden_dim)
        else:
            raise NotImplementedError
        
        self.low_pass_filter = low_pass_filter
        self.low_pass_fc  = nn.Linear(hidden_dim, hidden_dim)
        self.high_pass_fc = nn.Linear(hidden_dim, hidden_dim)

        if args.fusion_method == 'concat':
            self.fusion_fc = nn.Linear(2*hidden_dim, 2*hidden_dim)
            # self.fusion_param = nn.Parameter(torch.FloatTensor(2*hidden_dim, 2*hidden_dim))
            # self.last_layer = nn.Linear(2*hidden_dim, output_dim)
        elif args.fusion_method == 'add' or args.fusion_method == 'max':
            self.fusion_fc = nn.Linear(hidden_dim, hidden_dim)
            # self.fusion_param = nn.Parameter(torch.FloatTensor(hidden_dim, hidden_dim))
            # self.last_layer = nn.Linear(hidden_dim, output_dim)
        else:
            raise NotImplementedError
    
        

    def construct_high_pass_adj(self, H_0, adj):
        H_0_norm = F.normalize(H_0, p=2, dim=1)
        pair_wise_cos = torch.mm(H_0_norm, H_0_norm.t())
        pair_wise_cos = pair_wise_cos * (adj!=0)
        return F.relu(pair_wise_cos)
    

    def fusion(self, H_low, H_high):
        if self.args.fusion_method == 'add':
            H =  self.args.fusion_beta * H_low + (1-self.args.fusion_beta) * H_high
        elif self.args.fusion_method == 'concat':
            H = torch.cat((self.args.fusion_beta*H_low, (1-self.args.fusion_beta)*H_high), dim=1)
        elif self.args.fusion_method == 'max':
            H = torch.max(H_low, H_high)
        else:
            raise NotImplementedError
        return H
    

    def forward(self, X, adj):
        H_0 = self.first_layer(X)
        self.high_pass_adj = normalize_adj_torch(self.construct_high_pass_adj(H_0, adj))
        self.high_pass_filter = construct_filter(adj=self.high_pass_adj, 
                                                 l=self.args.high_pass_layers, 
                                                 alpha=self.args.high_pass_alpha)

        self.H_low = self.low_pass_fc(self.low_pass_filter @ H_0)
        self.H_high = self.high_pass_fc(self.high_pass_filter @ H_0)
        H = self.fusion(self.H_low, self.H_high)        

        H = H + self.args.fusion_gamma * self.fusion_fc(H)
        # H = self.last_layer(H)
        # H = F.softmax(H, dim=1)
        
        return H



class cluster_model(nn.Module):
    def __init__(self, method, node_num, hidden_dim, cluster_num, H):
        super(cluster_model, self).__init__()
        self.method = method
        self.node_num = node_num
        self.hidden_dim = hidden_dim
        self.cluster_num = cluster_num

        if method == 'mlp':
            self.fc = nn.Linear(hidden_dim, cluster_num).to(H.device)
            # self.cluster_assi = F.softmax(self.fc(H), dim=1)
        elif method == 'kmeans':
            init = self.kmeans(H)
            self.cluster_assi = nn.Parameter(init)
        elif method == 'random':
            init = torch.rand(node_num, cluster_num)
            self.cluster_assi = nn.Parameter(init)
        else:
            raise NotImplementedError
        
          
    def kmeans(self, H):
        cluster_ids,_ = k_means(H, self.cluster_num, device='gpu', distance='cosine')
        cluster = torch.zeros(H.shape[0], self.cluster_num).to(H.device)
        cluster[torch.arange(H.shape[0]), cluster_ids] = 1
        return cluster
    

    def forward(self, H):
        # 对clustering assignment matrix进行列归一化会导致每一项都很小
        # 改成了行归一化
        if self.method == 'mlp':
            return F.softmax(self.fc(H), dim=1)
        else:
            return F.softmax(self.cluster_assi, dim=1)


class low_pass_model(nn.Module):
    # H_0 = MLP(X) or H_0 = linear(X)
    # H = \sum_{i=1}^{L} \alpha_i A^i H_0 W
    def __init__(self, args, low_pass_filter, input_dim, hidden_dim):
        super(low_pass_model, self).__init__()
        self.args = args
        self.low_pass_filter = low_pass_filter
        if args.first_transformation=='mlp':
            self.lin1 = MLP([input_dim, hidden_dim])
        elif args.first_transformation=='linear':
            self.lin1 = nn.Linear(input_dim, hidden_dim)
        else:
            raise NotImplementedError
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, X):
        H_0 = self.lin1(X)
        H = self.low_pass_filter @ H_0
        H = self.lin2(H)
        return H



class pool_based_model(nn.Module):
    def __init__(self, args, input_dim, hidden_dim, cluster_num, low_pass_filter):
        super(pool_based_model, self).__init__()
        self.args = args
        if args.encoder == 'GCN':
            self.encoding = GCNConv(input_dim, hidden_dim)
        elif args.encoder == 'low_pass':
            self.encoding = low_pass_model(args, low_pass_filter, input_dim, hidden_dim)
        else:
            raise NotImplementedError    
        self.dmon_pool = DMoNPooling(hidden_dim, cluster_num)
    
    def forward(self, X, A, edge_index):
        if self.args.encoder == 'GCN':
            H = self.encoding(X, edge_index)
        elif self.args.encoder == 'low_pass':
            H = self.encoding(X)
        s, out, out_adj, spectral_loss, ortho_loss, cluster_loss = self.dmon_pool(H, A)
        # remove the dimension for graph level
        return s[0], out[0], out_adj[0], spectral_loss, ortho_loss, cluster_loss


def kmeans_loss_fn(H, C):
    cluster_centroid_embedding = C.T @ H
    loss = 0
    for i in range(C.shape[1]):
        loss += ((H - cluster_centroid_embedding[i])* C[:, i].unsqueeze(1)).norm(p=2, dim=1).sum()
    return loss/C.shape[0]/C.shape[1]


def kmeans_contrastive_loss_fn(H, C):
    # pos: node represen
    pass


def kmeans_centroid_loss_fn(H, C, args):
    # pos: node representation & its cluster centroid
    # neg: node representation & other cluster centroids
    cluster_centroid_embedding = C.T @ H
    loss = 0


def spectral_loss_fn(C, A):
    temp = torch.matmul(C.t(), A)
    return -torch.trace(torch.matmul(temp, C))


def reg_loss_fn(C, args):
    if args.reg_loss == 'orth':
        # || C^T C - I ||_2^F
        return torch.norm(torch.matmul(C.t(), C)-
                      torch.eye(C.shape[1]).to(C.device), p=2)
    elif args.reg_loss == 'col':
        # sqrt(K)/n ||sum C_i ||_F - 1
        return torch.norm(C.sum(dim=0), p=2) * np.sqrt(C.shape[1])/C.shape[0] -1
    elif args.reg_loss == 'sqrt':
        # -trace(sqrt(C^T C))
        return -torch.trace(torch.sqrt(torch.matmul(C.t(), C)+1e-15))
    else:
        raise NotImplementedError
    


def reconstruction_loss_distance(A, B):
    return torch.norm(A-B, p=2)


def reconstruction_loss_cosine(A, B, beta=1):
    A = F.normalize(A, p=2, dim=1)
    B = F.normalize(B, p=2, dim=1)
    return torch.pow(1 - torch.sum(A*B, dim=1), beta).sum()