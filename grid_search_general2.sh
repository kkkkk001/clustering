data=$1
device=$2
for norm in 1 0
do 
    for layer in 3 2
    do
        for alpha in 0.8 0.5 0.2
        do
            for S_alpha in 0.8 0.5 0.2
            do
                for ssg0 in 0.0001 0.0005 0.001
                do
                    for ssg1 in 0.0005 0.0001 0
                    do 
                        for ssg in 0.0001 0.0005 0.001
                        do 
                            python my_train_snapshot.py --dataset ${data} --epochs 500 --norm $norm --device ${device} \
                            --top_layers $layer --top_alpha $alpha --top_prop sgc --top_linear_trans 1 \
                            --attr_layers $layer --attr_alpha $S_alpha --attr_r 1 --attr_prop sgc --attr_linear_trans 1 \
                            --fusion_method add --fusion_beta 0.5 \
                            --xprop_layers $layer --xprop_alpha $alpha --cprop_layers $layer --cprop_alpha $alpha \
                            --kmeans_loss cen --loss_lambda_kmeans 0.02 --temperature 2 --loss_lambda_prop 1 --sharpening 1 \
                            --loss_lambda_SSG0 $ssg0 --loss_lambda_SSG1 $ssg1  --loss_lambda_SSG2 $ssg --loss_lambda_SSG3 $ssg \
                            --log_file res_log/${data}_grid_search2.txt --log_fold_file res_fold_log/${data}_grid_search2.txt
                        done
                    done
                done
            done
        done
    done
done