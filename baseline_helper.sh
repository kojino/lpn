#!/bin/bash
#SBATCH -o val2.%j.txt
#SBATCH -e val2.%j.err
#SBATCH -t 3-01:01:00
#SBATCH --mem 10000
#SBATCH -J crossval2
#SBATCH -p shared
#SBATCH --reservation=koshiba
python -m deeplp.baseline \
--data $1 \
--split_seed $2 \
--unlabel_prob $3
