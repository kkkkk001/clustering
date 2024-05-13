import torch
import torch.nn.functional as F

def cluster_id2assignment(cluster_ids, cluster_num):
    C = torch.zeros(cluster_ids.shape[0], cluster_num)
    C[torch.arange(cluster_ids.shape[0]), cluster_ids] = 1
    return C



def Cprop(C0, A, args):
    C = C0
    for _ in range(args.cprop_layers):
        C = args.cprop_alpha * torch.spmm(A, C) + C0
    return C

    


def t(C0, A, args):
    C = C0
    for _ in range(args.low_pass_layers):
        C = args.alphaC * torch.spmm(A, C) + C0
    return C