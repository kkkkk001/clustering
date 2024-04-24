from dgc.utils import load_graph_data, normalize_adj, construct_filter, normalize_adj_torch
from dgc.clustering import k_means
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
from torch_geometric.nn import DMoNPooling, GCNConv

from torch_geometric.nn.models.mlp import MLP
from typing import Optional, Tuple
from torch_geometric.nn import Linear
from math import log
from torch.nn import ModuleList
from torch import Tensor

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import GCNConv


from torch_geometric.typing import Adj, OptTensor
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch.nn import Parameter
from torch_geometric.nn.conv.gcn_conv import gcn_norm


class DeProp_Prop(MessagePassing):

    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, in_channels: int, out_channels: int, lambda1: float, lambda2: float, gamma: float,
                 improved: bool = False,  cached: bool = True,
                 add_self_loops: bool = True, normalize: bool = True,
                 bias: bool = True, **kwargs):

        super(DeProp_Prop, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.gamma = gamma
        self._cached_edge_index = None
        self._cached_adj_t = None
        self.lin = Linear(in_channels, out_channels, bias=False,
                          weight_initializer='glorot')

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()


    def reset_parameters(self):
        self.lin.reset_parameters()
        zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None


    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        x = self.lin(x)
        z = (1 - self.gamma * self.lambda1 + self.gamma * self.lambda2) * x
        s = self.gamma * self.lambda1 * self.propagate(edge_index, x=x, edge_weight=edge_weight,
                              size=None)
        t = self.gamma * self.lambda2 * x @ (x.t() @ x)
        out = z + s - t

        if self.bias is not None:
            out += self.bias

        return out


    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j


    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)


    def __repr__(self):
        return '{}(lambda1={}, lambda2={}, gamma={})'.format(
            self.__class__.__name__, self.lambda1, self.lambda2, self.gamma)


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, n_layers=2):
        super().__init__()
        self.fc = nn.Linear(in_channels, hidden_channels)
        self.convs = nn.ModuleList()
        for _ in range(n_layers-1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.last_conv = GCNConv(hidden_channels, hidden_channels)
        

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc(x).relu()
        for conv in self.convs:
            x = F.dropout(x, p=0.5, training=self.training)
            x = conv(x, edge_index, edge_weight).relu()

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.last_conv(x, edge_index, edge_weight)
        return x


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



class encoding2(nn.Module):
    def __init__(self, args, input_dim, hidden_dim, norm_adj, cluster_num):
        super(encoding2, self).__init__()
        self.args = args

        if args.first_transformation=='mlp':
            self.first_layer = nn.Sequential(nn.Linear(input_dim, hidden_dim), 
                                             nn.ReLU(),
                                             nn.Linear(hidden_dim, hidden_dim))
        elif args.first_transformation=='linear':
            self.first_layer = nn.Linear(input_dim, hidden_dim)
        else:
            raise NotImplementedError
        
        self.norm_adj = norm_adj
        self.cluster_num = cluster_num

        self.lins = ModuleList()
        for i in range(self.args.low_pass_layers):
            self.lins.append(Linear(in_channels = hidden_dim, out_channels = hidden_dim, bias = True, weight_initializer = 'glorot'))

        self.low_pass_fc  = nn.Linear(hidden_dim, hidden_dim)
        self.high_pass_fc = nn.Linear(hidden_dim, hidden_dim)

        self.final_layer = Linear(in_channels = hidden_dim, out_channels = hidden_dim, bias = True, weight_initializer = 'glorot')

        self.cluster_layer = Linear(in_channels = hidden_dim, out_channels = self.cluster_num, bias = True, weight_initializer = 'glorot')
        #nn.Sequential(nn.Linear(hidden_dim, hidden_dim), 
                                             # nn.ReLU(),
                                             # nn.Linear(hidden_dim, self.cluster_num))

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
    
        

    def construct_high_pass_adj(self, H_0):
        H_0_norm = F.normalize(H_0, p=2, dim=1)
        pair_wise_cos = torch.mm(H_0_norm, H_0_norm.t())
        pair_wise_cos = pair_wise_cos * (self.norm_adj!=0)
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
    

    def forward(self, X, mask: Optional[Tensor] = None):
        H_0 = self.first_layer(X)
        # self.high_pass_adj = normalize_adj_torch(self.construct_high_pass_adj(H_0))
        # self.high_pass_filter = construct_filter(adj=self.high_pass_adj, 
                                                 # l=self.args.high_pass_layers, 
                                                 # alpha=self.args.high_pass_alpha)
        H = H_0
        for i in range(self.args.low_pass_layers):
            # temp = torch.mm(H_0.t(), H)
            # temp = torch.mm(H_0, temp)
            # DeProp(H, H_0, self.norm_adj, self.args.step_size_gamma, self.args.alphaH, self.args.alphaO)
            # H = F.normalize(H, p=2, dim=1)

            # H = self.args.alphaH * torch.spmm(self.norm_adj, H) + H_0
            # H = (1-self.args.alphaH) * torch.spmm(self.norm_adj, H) + self.args.alphaH * H_0
            H = DePropagate(H, H_0, self.norm_adj, self.args.step_size_gamma, self.args.alphaH, self.args.alphaO)
            # H = F.normalize(H, p=2, dim=1)



            # H_low = self.alpha*self.norm_adj(H_low) + H_0

            # theta = log(self.args.gamma / (i + 2))
            # H = (1 - theta) * H + theta * self.lins[i].forward(H)
            # H = H + self.lins[i].forward(H)
            # H = F.dropout(H, p = self.args.dropout, training = self.training, inplace = True)
            # H = F.relu(H, inplace = True)
            # H = F.selu(H, inplace = True)

        # self.H_high = self.high_pass_fc(self.high_pass_filter @ H_0)
        # H = self.fusion(self.H_low, self.H_high)        

        # H = H + self.args.fusion_gamma * self.fusion_fc(H)
        H = self.final_layer(H)
        # H = F.dropout(H, p = self.args.dropout, training = self.training, inplace = True)
        H = F.normalize(H, p=2, dim=1)

        
        return H


class encoding3(nn.Module):
    def __init__(self, args, input_dim, hidden_dim, norm_adj, cluster_num):
        super(encoding3, self).__init__()
        self.args = args
        self.norm_adj = norm_adj
        self.cluster_num = cluster_num
        self.convs = nn.ModuleList()
        self.convs.append(DeProp_Prop(input_dim, hidden_dim, args.alphaH, args.alphaO, args.step_size_gamma))
        for _ in range(self.args.low_pass_layers-1):
            self.convs.append(DeProp_Prop(hidden_dim, hidden_dim, args.alphaH, args.alphaO, args.step_size_gamma))
        self.fc = nn.Linear(hidden_dim, cluster_num)

        self.dropout = args.dropout

    def forward(self, x, edge_index, mask: Optional[Tensor] = None):
        for i, conv in enumerate(self.convs):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, edge_index)
            x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc(x)

        x = F.dropout(x, p=self.dropout, training=self.training)
        return F.log_softmax(x, dim=1)



    

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


def kmeans_loss_fn(H, C, args):
    if args.kmeans_loss == 'tr':
        return kmeans_trace_loss_fn(H, C)
    elif args.kmeans_loss == 'cen':
        return kmeans_centroid_contrastive_loss_fn(H, C, args)
    elif args.kmeans_loss == 'nod':
        return kmeans_node_contrastive_loss_fn(H, C)
    else:
        raise NotImplementedError


def kmeans_trace_loss_fn(H, C):
    cluster_centroid_embedding = C.T @ H
    loss = 0
    for i in range(C.shape[1]):
        loss += ((H - cluster_centroid_embedding[i])* C[:, i].unsqueeze(1)).norm(p=2, dim=1).sum()
    return loss/C.shape[0]/C.shape[1]


def kmeans_centroid_contrastive_loss_fn(H, C, args):
    # pos: node representation vs. its cluster centroid
    # neg: node representation vs. other cluster centroids
    cluster_centroid_embedding = C.T @ H
    sim = torch.einsum("nd, cd -> nc", H, cluster_centroid_embedding)
    sim /= args.temperature
    labels = torch.argmax(C, dim=1)
    return F.cross_entropy(sim, labels) 

# TODO: implement this function
def density_estimation(H, C, args):
    return
    # concentration estimation (phi)        
    density = np.zeros(k)
    for i,dist in enumerate(Dcluster):
        if len(dist)>1:
            d = (np.asarray(dist)**0.5).mean()/np.log(len(dist)+10)            
            density[i] = d     
            
    #if cluster only has one point, use the max to estimate its concentration        
    dmax = density.max()
    for i,dist in enumerate(Dcluster):
        if len(dist)<=1:
            density[i] = dmax 

    density = density.clip(np.percentile(density,10),np.percentile(density,90)) #clamp extreme values for stability
    density = args.temperature*density/density.mean()  #scale the mean to temperature 

def kmeans_node_contrastive_loss_fn(H, C, num_neg_samples):
    # pos: nodes in the same cluster
    # neg: nodes in different clusters

    labels = torch.argmax(C, dim=1)
    num_nodes = H.shape[0]
    loss = 0

    for i in range(num_nodes):
        target_label = labels[i]
        pos_indices = torch.where(labels == target_label)[0]
        neg_indices = torch.where(labels != target_label)[0]

        # 随机选择一个正样本
        pos_index = pos_indices[torch.randperm(pos_indices.size(0))[0]]
        pos_sample = H[pos_index]

        # 随机选择 num_neg_samples 个负样本
        neg_indices = neg_indices[torch.randperm(neg_indices.size(0))[:num_neg_samples]]
        neg_samples = H[neg_indices]

        # 计算 InfoNCE loss
        pos_sim = torch.dot(H[i], pos_sample)
        neg_sim = torch.matmul(H[i], neg_samples.T)
        logits = torch.cat([pos_sim.unsqueeze(0), neg_sim])
        targets = torch.zeros(logits.shape[0], dtype=torch.long, device=H.device)
        loss += F.cross_entropy(logits, targets)

    return loss / num_nodes


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


def DePropagate(C, C0, A, gamma, alphaC, alphaO):
    z =  (1 - gamma * alphaC + gamma * alphaO) * C
    s = gamma * alphaC * torch.spmm(A, C)
    t = gamma * alphaO * C @ (C.t() @ C)
    return z + s - t + gamma*C0