# for Hlr in 0.01 0.005
# do 
#     for step_size in 0.005 0.01
#     do 
#         for lambda1 in 60 100 140 180 220 260 300
#         do 
#             for lambda2 in 0.001 0.003 0.005 0.01
#             do 
#             python my_train3.py --dataset $1 --H_lr $Hlr --gnnlayers 4 --loss_lambda_kmeans 0 --step_size_gamma $step_size --lambda1 $lambda1 --lambda2 $lambda2 --dropout 0 --log_file $2
#             done
#         done
#     done


# for top_layers in 5 4 3 2
# do
#     for top_alpha in 1 0.8 0.5 0.2
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
#                                 python my_train4.py --dataset texas --t_lr 0.01 --a_lr 0.01 --epochs 300  --top_layers $top_layers --top_alpha $top_alpha --attr_layers $attr_layers --attr_alpha $attr_alpha --attr_r 1 --fusion_beta 0.2 --xprop_layers $xprop_layers --xprop_alpha $xprop_alpha --cprop_layers $cprop_layers --cprop_alpha $cprop_alpha --loss_lambda_kmeans 0.01 --log_file res_log/grid_search_ssg5.txt --loss_lambda_SSG0 0 --loss_lambda_SSG1 0 --loss_lambda_SSG2 0 --loss_lambda_SSG3 0
#                             done
#                         done
#                     done
#                 done
#             done
#         done
#     done
# done



# for layer in 5 4
# do
#     for top_alpha in 1 0.8 0.5 0.2
#     do
#         for attr_alpha in 1 0.8 0.5 0.2
#         do
#             for xprop_alpha in 1 0.8 0.5 0.2
#             do
#                 for cprop_alpha in 1 0.8 0.5 0.2
#                 do
#                     # python my_train5.py --dataset texas \
#                     # --top_layers $layer --top_alpha $top_alpha --top_prop sgc --top_linear_trans 1 \
#                     # --attr_layers $layer --attr_alpha $attr_alpha --attr_r 1 --attr_prop sgc --attr_linear_trans 1 \
#                     # --fusion_method add --fusion_beta 0.5 \
#                     # --xprop_layers $layer --xprop_alpha $xprop_alpha --cprop_layers $layer --cprop_alpha $cprop_alpha \
#                     # --kmeans_loss cen --loss_lambda_kmeans 0.01 --temperature 2 --loss_lambda_prop 1 --sharpening 1 \
#                     # --loss_lambda_SSG0 0.001 --loss_lambda_SSG3 0.001 \
#                     # --log_file res_log/grid_search_ssg8.txt 
#                     python my_train5.py --dataset cora --epochs 500 \
#                     --top_layers $layer --top_alpha $top_alpha --top_prop sgc --top_linear_trans 1 \
#                     --attr_layers $layer --attr_alpha $attr_alpha --attr_r 1 --attr_prop sgc --attr_linear_trans 1 \
#                     --fusion_method add --fusion_beta 0.5 \
#                     --xprop_layers $layer --xprop_alpha $xprop_alpha --cprop_layers $layer --cprop_alpha $cprop_alpha \
#                     --kmeans_loss cen --loss_lambda_kmeans 0 --temperature 2 --loss_lambda_prop 1 --sharpening 1 \
#                     --loss_lambda_SSG0 0.001 --loss_lambda_SSG1 0 --loss_lambda_SSG2 0.001 --loss_lambda_SSG3 0.001 \
#                     --log_file res_log/grid_search_ssg10.txt 
#                 done
#             done
#         done
#     done
# done


for layer in 5 4 3 2
do
    for alpha in 1 0.8 0.5 0.2
    do
        for S_alpha in 1 0.8 0.5 0.2
        do
            for ssg0 in 0 0.0001 0.0005 0.001
            do
                for ssg in 0.001 0.0005 0.0001 0
                do 
                    python my_train_snapshot2.py --dataset texas --epochs 500 \
                    --top_layers $layer --top_alpha $alpha --top_prop sgc --top_linear_trans 1 \
                    --attr_layers $layer --attr_alpha $S_alpha --attr_r 1 --attr_prop sgc --attr_linear_trans 1 \
                    --fusion_method add --fusion_beta 0.5 \
                    --xprop_layers $layer --xprop_alpha $alpha --cprop_layers $layer --cprop_alpha $alpha \
                    --kmeans_loss cen --loss_lambda_kmeans 0.02 --temperature 2 --loss_lambda_prop 1 --sharpening 1 \
                    --loss_lambda_SSG0 $ssg0 --loss_lambda_SSG1 $ssg  --loss_lambda_SSG2 $ssg --loss_lambda_SSG3 $ssg \
                    --log_file res_log/grid_search_ssg10.txt 
                done
            done
        done
    done
done