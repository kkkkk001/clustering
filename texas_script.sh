python my_train7.py --dataset texas --epochs 500 --svd_on_A A --fusion_norm none --norm 1 \
--top_layers 5 --top_alpha 0.2 --top_prop sgc --top_linear_trans mlp \
--attr_layers 5 --attr_alpha 0.5 --attr_r 1 --attr_prop sgc --attr_linear_trans mlp \
--fusion_method add --fusion_beta 0.5 \
--xprop_layers 5 --xprop_alpha 0.2 --cprop_layers 1 --cprop_alpha 1 \
--kmeans_loss cen --loss_lambda_kmeans 0.02 --temperature 2 --loss_lambda_prop 1 --sharpening 1 \
--loss_lambda_SSG0 0.0001 --loss_lambda_SSG1 0.006  --loss_lambda_SSG2 0.006 --loss_lambda_SSG3 0.006 

