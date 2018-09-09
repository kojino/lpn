#!/bin/bash
#SBATCH -o val2.%j.txt
#SBATCH -e val2.%j.err
#SBATCH -t 5-05:01:00
#SBATCH --mem 3999
#SBATCH -J edge_master
#SBATCH -p shared


python -m deeplp.main \
--batch_size=${1} \
--bifurcation=${2} \
--crossval_k=${3} \
--data=${4} \
--decay=${5} \
--feature_type=${6} \
--lamda=${7} \
--leave_k=${8} \
--log=${9} \
--lr=${10} \
--model=${11} \
--num_epoch=${12} \
--num_layers=${13} \
--num_samples=${14} \
--save_params=${15} \
--split_seed=${16} \
--unlabel_prob=${17} \
--weighted_loss=${18} 
