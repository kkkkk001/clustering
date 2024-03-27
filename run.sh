# for data in cora citeseer amap bat eat uat texas wisc cornell actor 
for data in cora 
do
    python my_train.py --dataset $data --cluster_init_method mlp --fusion_method concat
    # python my_train.py --dataset $data --cluster_init_method mlp --fusion_method max
    # python my_train.py --dataset $data --cluster_init_method mlp --fusion_gamma 0.5
    # python my_train.py --dataset $data --cluster_init_method mlp --fusion_gamma 2
    python my_train.py --dataset $data --cluster_init_method mlp --reg_loss col
    python my_train.py --dataset $data --cluster_init_method mlp --reg_loss sqrt
    python my_train.py --dataset $data --cluster_init_method kmeans
    python my_train.py --dataset $data --cluster_init_method random


done



# for data in cora citeseer amap bat eat uat texas wisc cornell actor 
# for data in texas wisc cornell actor
# do
#     python my_train.py --dataset $data --cluster_init_method mlp --fusion_beta 0.2
# done