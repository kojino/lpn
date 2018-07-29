#!/bin/bash
#SBATCH -o val2.%j.txt
#SBATCH -e val2.%j.err
#SBATCH -t 3-01:01:00
#SBATCH --mem 3999
#SBATCH -J crossval2
#SBATCH -p shared
#SBATCH --reservation=koshiba
python seed_features.py \
--data $1 \
--seed $2 \
--source $3
