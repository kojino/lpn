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
for seed in ${seeds[@]}
do
./master_helper.sh \
$data \
$seed 
done
done
echo "end"
