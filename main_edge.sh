#!/bin/bash
#SBATCH -o val1.%j.txt
#SBATCH -e val1.%j.err
#SBATCH -t 05:01:00
#SBATCH --mem 3999
#SBATCH -J edge_master
#SBATCH -p shared

batch_sizes=(400)
bifurcations=(0.001)
crossval_ks=(5 1)
datas=('linqs_pubmed_planetoid' 'linqs_cora_planetoid' 'linqs_citeseer_planetoid')
decays=(0)
feature_types=('all')
keep_probs=(1.0)
lamdas=(-200 -14 -12 -10 -8 -6 -4 -2 0 2) 
leave_ks=(1)
log='DEBUG'
logistics=(1)
lrs=(0.001)
models=('edge')
num_epochs=(2000)
num_layerss=(-1)
num_sampless=(400)
save_params=1
settings=('planetoid_random')
split_seeds=(0 1 2 3 4 5 6 7 8 9)
unlabel_probs=(0.99)
weighted_loss=1

for setting in ${settings[@]}
do
for model in ${models[@]}
do
for data in ${datas[@]}
do
for feature_type in ${feature_types[@]}
do
for crossval_k in ${crossval_ks[@]}
do
for unlabel_prob in ${unlabel_probs[@]}
do
for num_layers in ${num_layerss[@]}
do
for split_seed in ${split_seeds[@]}
do
for keep_prob in ${keep_probs[@]}
do
for num_epoch in ${num_epochs[@]}
do
for leave_k in ${leave_ks[@]}
do
for lr in ${lrs[@]}
do
for decay in ${decays[@]}
do
for batch_size in ${batch_sizes[@]}
do
for lamda in ${lamdas[@]}
do
for bifurcation in ${bifurcations[@]}
do
for num_samples in ${num_sampless[@]}
do
for logistic in ${logistics[@]}
do
./main_helper.sh \
$batch_size \
$bifurcation \
$crossval_k \
$data \
$decay \
$feature_type \
$keep_prob \
$lamda \
$leave_k \
$log \
$logistic \
$lr \
$model \
$num_epoch \
$num_layers \
$num_samples \
$save_params \
$setting \
$split_seed \
$unlabel_prob \
$weighted_loss 
done
done
done
done
done
done
done
done
done
done
done
done
done
done
done
done
done
done
echo "end"
