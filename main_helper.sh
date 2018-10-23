#!/bin/bash
#SBATCH -o val2.%j.txt
#SBATCH -e val2.%j.err
#SBATCH -t 5-05:01:00
#SBATCH --mem 3999
#SBATCH -J edge_master
#SBATCH -p general, shared

python -m deeplp.main \
--batch_size=${1} \
--bifurcation=${2} \
--crossval_k=${3} \
--data=${4} \
--decay=${5} \
--feature_type=${6} \
--keep_prob=${7} \
--lamda=${8} \
--leave_k=${9} \
--log=${10} \
--lr=${11} \
--model=${12} \
--num_epoch=${13} \
--num_layers=${14} \
--num_samples=${15} \
--save_params=${16} \
--setting=${17} \
--split_seed=${18} \
--unlabel_prob=${19} \
--weighted_loss=${20} 
