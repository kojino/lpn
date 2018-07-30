#!/bin/bash
#SBATCH -p shared
#SBATCH -o val1.%j.txt
#SBATCH -e val1.%j.err
#SBATCH -t 05:01:00
#SBATCH --mem 1000
#SBATCH -J crossval1
#SBATCH --reservation=koshiba

datas=('cora')
split_seeds=(0 1 2 3 4 5 6 7 8 9)
unlabel_probs=(0.99)
for data in ${datas[@]}
do
for split_seed in ${split_seeds[@]}
do
for unlabel_prob in ${unlabel_probs[@]}
do
./baseline_helper.sh $data $split_seed $unlabel_prob
done
done
done
echo "end"
