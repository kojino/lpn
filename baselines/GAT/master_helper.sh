#!/bin/bash
#SBATCH -o val2.%j.txt
#SBATCH -e val2.%j.err
#SBATCH -t 3-01:01:00
#SBATCH --mem 10000
#SBATCH -J crossval2
#SBATCH -p shared
python execute_cora_sparse.py \
--data $1 \
--seed $2 \
--unlabel_prob $3
