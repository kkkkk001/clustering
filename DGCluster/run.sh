datasets="texas cornell wisc actor squirrel chameleon crocodile cora citeseer pubmed "
lams=0
seeds=0

for dataset in $datasets; do
  for lam in $lams; do
    for seed in $seeds; do
      python -u main.py --dataset $dataset --lam $lam --seed $seed --device cuda:0 > log/$dataset.log
    done
  done
done