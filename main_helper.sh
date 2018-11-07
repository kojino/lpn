#!/bin/bash
#SBATCH -o val2.%j.txt
#SBATCH -e val2.%j.err
#SBATCH -t 5-05:01:00
#SBATCH --mem 2999
#SBATCH -J edge_master
#SBATCH -p serial_requeue
#SBATCH --account=ysinger_group

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
--logistic=${11} \
--lr=${12} \
--model=${13} \
--num_epoch=${14} \
--num_layers=${15} \
--num_samples=${16} \
--save_params=${17} \
--setting=${18} \
--split_seed=${19} \
--unlabel_prob=${20} \
--weighted_loss=${21} 
