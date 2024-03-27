# -*- coding: utf-8 -*-
# @Author  : Yue Liu
# @Email   : yueliu19990731@163.com
# @Time    : 2021/11/25 11:11

import torch
import random
import numpy as np
from .kmeans_gpu import kmeans
from sklearn.cluster import KMeans


def k_means(embedding, k, device="cpu", distance="euclidean"):
    """
    K-means algorithm
    :param embedding: embedding of clustering
    :param k: hyper-parameter in K-means
    :param y_true: ground truth
    :param device: device
    :returns acc, nmi, ari, f1, center:
    - acc
    - nmi
    - ari
    - f1
    - cluster centers
    """
    if device == "cpu":
        model = KMeans(n_clusters=k, n_init=20)
        cluster_id = model.fit_predict(embedding)
        center = model.cluster_centers_
    if device == "gpu":
        # if embedding is not torch tensor, convert it to torch tensor
        if not isinstance(embedding, torch.Tensor):
            embedding = torch.tensor(embedding).to("cuda")
        cluster_id, center, _ = kmeans(X=embedding, num_clusters=k, distance=distance, device="cuda")
        cluster_id = cluster_id.numpy()
    return cluster_id, center
