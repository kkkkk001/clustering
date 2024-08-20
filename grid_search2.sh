for layer in 4 5
do
    for fusion in 0.2 0.8 
    do
        for ssg0 in 0.0006 0.001  
        do
            for ssg in 0.006 0.01 
            do 
                python my_train_snapshot.py --dataset cornell --epochs 500 --norm 1 \
                --top_layers $layer --top_alpha 0.2 --top_prop sgc --top_linear_trans 1 \
                --attr_layers $layer --attr_alpha 0.5 --attr_r 1 --attr_prop sgc --attr_linear_trans 1 \
                --fusion_method add --fusion_beta $fusion \
                --xprop_layers $layer --xprop_alpha 0.2 --cprop_layers $layer --cprop_alpha 0.2 \
                --kmeans_loss cen --loss_lambda_kmeans 0.02 --temperature 2 --loss_lambda_prop 1 --sharpening 1 \
                --loss_lambda_SSG0 $ssg0 --loss_lambda_SSG1 $ssg  --loss_lambda_SSG2 $ssg --loss_lambda_SSG3 0$ssg 
            done
        done
    done 
done
