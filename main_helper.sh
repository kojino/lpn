#!/bin/bash
#SBATCH -o val2.%j.txt
#SBATCH -e val2.%j.err
#SBATCH -t 06:50
#SBATCH --mem 3999
#SBATCH -J main
#SBATCH -p seas_dgx1 
#SBATCH --gres=gpu -n 4

# module load cuda/9.0-fasrc02 cudnn/7.0_cuda9.0-fasrc01
# module load Anaconda3/5.0.1-fasrc01
# source activate tf1.8_cuda9

python -m deeplp.main \
--change_b $1 \
--clamp $2 \
--crossval_k $3 \
--data $4 \
--has_features $5 \
--leave_k $6 \
--logistic $7 \
--loss_class_mass $8 \
--lr $9 \
--model ${10} \
--num_epoch ${11} \
--num_iter ${12} \
--num_samples ${13} \
--parameter_seed ${14} \
--regularize_theta ${15} \
--regularize_type ${16} \
--split_seed ${17} \
--asymmetric ${18} \
--unlabel_prob ${19} \
--weight_normalization ${20} \
--confidence ${21} \
--datatype ${22} \
--profile ${23} \
--sparse_edges ${24} \
--regularize_weight ${25} \
--clamp_scale ${26}
