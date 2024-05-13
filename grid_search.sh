# for Hlr in 0.01 0.005
# do 
#     for step_size in 0.005 0.01
#     do 
#         for lambda1 in 60 100 140 180 220 260 300
#         do 
#             for lambda2 in 0.001 0.003 0.005 0.01
#             do 
#             python my_train3.py --dataset $1 --H_lr $Hlr --gnnlayers 4 --loss_lambda_kmeans 0 --step_size_gamma $step_size --lambda1 $lambda1 --lambda2 $lambda2 --dropout 0 
#             done
#         done
#     done                  
# done
for cprop_layers in 3 4 5
do
    for cprop_alpha in 0.2 0.5 0.8 1
    do
        for lmd in 0.01 0.005
        do 
            for bt in 0.8 1.0
            do
                python my_train3.py --H_lr=0.01 --loss_lambda_kmeans $lmd --gnnlayers=4 --with_bn=True --F_norm=True --lin=True --step_size_gamma=0.01 --lambda1=180.0 --lambda2=0.003 --dropout=0.0 --dataset cora --fusion_beta 1.0 --epochs 300 --loss_lambda_SSG 0 --kmeans_loss tr --cprop_layers $cprop_layers --cprop_alpha $cprop_alpha --xprop_layers 5 --xprop_alpha 1 
            done
        done
    done
done


