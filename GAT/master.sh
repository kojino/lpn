#!/bin/bash
#SBATCH -o val1.%j.txt
#SBATCH -e val1.%j.err
#SBATCH -t 05:01:00
#SBATCH --mem 3999
#SBATCH -J edge_master
#SBATCH -p shared

datas=('cora' 'citeseer' 'pubmed')
seeds=(0 1 2 3 4 5 6 7 8 9)

for data in ${datas[@]}
do
    if [ $data = "pubmed" ]; then
    unlabel_probs=(0.9975 0.995 0.9925 0.9)
    else
    unlabel_probs=(0.99 0.98 0.97 0.96 0.95)
    fi
    for seed in ${seeds[@]}
    do
        for unlabel_prob in ${unlabel_probs[@]}
        do
            ./master_helper.sh \
            $data \
            $seed \
            $unlabel_prob
        done
    done
done
echo "end"
