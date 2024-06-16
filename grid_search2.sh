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


for top_layers in 5 4 3 2
do
    for top_alpha in 1 0.8 0.5 0.2
    do
        for attr_layers in 2 3 4 5
        do
            for attr_alpha in 0.2 0.5 0.8 1
            do
                for xprop_layers in 2 3 4 5
                do   
                    for xprop_alpha in 0.2 0.5 0.8 1
                    do
                        for cprop_layers in 2 3 4 5
                        do
                            for cprop_alpha in 0.2 0.5 0.8 1
                            do
                                python my_train4.py --dataset texas --t_lr 0.01 --a_lr 0.01 --epochs 300  --top_layers $top_layers --top_alpha $top_alpha --attr_layers $attr_layers --attr_alpha $attr_alpha --attr_r 1 --fusion_beta 0.2 --xprop_layers $xprop_layers --xprop_alpha $xprop_alpha --cprop_layers $cprop_layers --cprop_alpha $cprop_alpha --loss_lambda_kmeans 0.01 --log_file res_log/grid_search_ssg5.txt --loss_lambda_SSG0 0 --loss_lambda_SSG1 0 --loss_lambda_SSG2 0 --loss_lambda_SSG3 0
                            done
                        done
                    done
                done
            done
        done
    done
done