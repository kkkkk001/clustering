link to datasets: https://drive.google.com/file/d/1J5IuMV40CETvOs0wS3HqI0929laU22G0/view?usp=sharing
Please download and put the folders of datasets in cluster/dataset, e.g., cluster/dataset/cora/

test dmon with GCN encoder:

python test_dmon.py --encoder GCN

test dmon with our low pass filter encoder:

python test_dmon.py --encoder low_pass


test DGCluster:

cd DGClustter
python mian.py --dataset cora --device cuda:0 --lam 0 --encoder GCN
python mian.py --dataset cora --device cuda:0 --lam 0 --encoder low_pass