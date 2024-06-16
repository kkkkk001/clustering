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
# for cprop_layers in 3 4 5
# do
#     for cprop_alpha in 0.2 0.5 0.8 1
#     do
#         for lmd in 0.01 0.005
#         do 
#             for bt in 0.8 1.0
#             do
#                 python my_train3.py --H_lr=0.01 --loss_lambda_kmeans $lmd --gnnlayers=4 --with_bn=True --F_norm=True --lin=True --step_size_gamma=0.01 --lambda1=180.0 --lambda2=0.003 --dropout=0.0 --dataset cora --fusion_beta 1.0 --epochs 300 --loss_lambda_SSG 0 --kmeans_loss tr --cprop_layers $cprop_layers --cprop_alpha $cprop_alpha --xprop_layers 5 --xprop_alpha 1 
#             done
#         done
#     done
# done






# for loss_lambda_kmeans in 0.005 0.01 0.02 0.03
# do
#     for cprop_layers in 3 4 5
#     do
#         for cprop_alpha in 0.2 0.5 0.8 1
#         do
#             for ssg0 in 0.0005 0.001 0.002 
#             do 
#                 for ssg1 in 0.005 0.01 0.02
#                 do
#                     for ssg2 in 0.005 0.01 0.02
#                     do
#                         for ssg3 in 0.005 0.01 0.02
#                         do
#                             python my_train3.py --dataset texas --H_lr 0.005 --epochs 100  --attr_layers 5 --attr_alpha 1 --attr_r 1 --fusion_beta 0.5 --xprop_layers 5 --xprop_alpha 0.2  --with_bn 1 --gnnlayers 4 --step_size_gamma 0.01 --lambda1 300.0 --lambda2 0.003  --loss_lambda_SSG0 $ssg0 --loss_lambda_SSG1 $ssg1 --loss_lambda_SSG2 $ssg2 --loss_lambda_SSG3 $ssg3 --loss_lambda_kmeans $loss_lambda_kmeans --cprop_layers $cprop_layers --cprop_alpha $cprop_alpha --log_file res_log/grid_search_ssg1.txt
#                         done
#                     done
#                 done
#             done
#         done
#     done
# done

# for top_layers in 2 3 4 5
# do
#     for top_alpha in 0.2 0.5 0.8 1
#     do
#         for attr_layers in 2 3 4 5
#         do
#             for attr_alpha in 0.2 0.5 0.8 1
#             do
#                 for xprop_layers in 2 3 4 5
#                 do   
#                     for xprop_alpha in 0.2 0.5 0.8 1
#                     do
#                         for cprop_layers in 2 3 4 5
#                         do
#                             for cprop_alpha in 0.2 0.5 0.8 1
#                             do
#                                 python my_train4.py --dataset texas --t_lr 0.01 --a_lr 0.01 --epochs 300  --top_layers $top_layers --top_alpha $top_alpha --attr_layers $attr_layers --attr_alpha $attr_alpha --attr_r 1 --fusion_beta 0.2 --xprop_layers $xprop_layers --xprop_alpha $xprop_alpha --cprop_layers $cprop_layers --cprop_alpha $cprop_alpha --loss_lambda_kmeans 0.01 --log_file res_log/grid_search_ssg4.txt --loss_lambda_SSG0 0 --loss_lambda_SSG1 0 --loss_lambda_SSG2 0 --loss_lambda_SSG3 0
#                             done
#                         done
#                     done
#                 done
#             done
#         done
#     done
# done


# for A_layers in 2 3 4 5
# do
#     for A_alpha in 0.2 0.5 0.8 1
#     do
#         for X_layers in 2 3 4 5
#         do
#             for X_alpha in 0.2 0.5 0.8 1
#             do
#                 python my_train4.py --dataset texas --t_lr 0.01 --a_lr 0.01 --epochs 300  --top_layers $A_layers --top_alpha $A_alpha --attr_layers $X_layers --attr_alpha $X_alpha --attr_r 1 --fusion_beta 0.2 --xprop_layers $A_layers --xprop_alpha $A_alpha --cprop_layers $A_layers --cprop_alpha $A_alpha --loss_lambda_kmeans 0.02 --log_file res_log/grid_search_ssg6.txt --loss_lambda_SSG0 0 --loss_lambda_SSG1 0 --loss_lambda_SSG2 0 --loss_lambda_SSG3 0
#             done
#         done
#     done
# done


for ssg0 in 0 0.0001 0.0005 0.001
do 
    for ssg1 in 0 0.01 0.02 0.05
    do
        for ssg2 in 0 0.01 0.02 0.05
        do
            for ssg3 in 0 0.001 0.005 0.01
            do
                python my_train4.py --dataset texas --t_lr 0.01 --a_lr 0.01 --epochs 300  --top_layers 2 --top_alpha 0.2 --attr_layers 5 --attr_alpha 1.0 --attr_r 1 --fusion_beta 0.2 --xprop_layers 2 --xprop_alpha 0.2 --cprop_layers 2 --cprop_alpha 0.2 --loss_lambda_kmeans 0.01 --log_file res_log/grid_search_ssg7.txt --loss_lambda_SSG0 $ssg0 --loss_lambda_SSG1 $ssg1 --loss_lambda_SSG2 $ssg2 --loss_lambda_SSG3 $ssg3
            done
        done
    done
done
