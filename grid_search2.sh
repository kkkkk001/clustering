for Hlr in 0.01 0.005
do 
    for step_size in 0.005 0.01
    do 
        for lambda1 in 60 100 140 180 220 260 300
        do 
            for lambda2 in 0.001 0.003 0.005 0.01
            do 
            python my_train3.py --dataset $1 --H_lr $Hlr --gnnlayers 4 --loss_lambda_kmeans 0 --step_size_gamma $step_size --lambda1 $lambda1 --lambda2 $lambda2 --dropout 0 --log_file $2
            done
        done
    done