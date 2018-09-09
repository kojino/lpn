#!/bin/bash
#SBATCH -o val1.%j.txt
#SBATCH -e val1.%j.err
#SBATCH -t 05:01:00
#SBATCH --mem 3999
#SBATCH -J edge_master
#SBATCH -p shared

batch_sizes=(130 50 10)
bifurcations=(0.001)
crossval_ks=(1)
datas=('linqs_cora')
decays=(0 10 100)
feature_types=('all')
lamdas=(-14 -12 -10 -8 -6 -4 -2 0)
leave_ks=(1)
log='DEBUG'
lrs=(0.1 0.01)
models=('edge')
num_epochs=(3000)
num_layerss=(30)
num_sampless=(1000)
save_params=1
split_seeds=(0 1 2 3 4 5)
unlabel_probs=(0.90 0.95 0.99)
weighted_loss=1

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
./main_helper.sh \
$batch_size \
$bifurcation \
$crossval_k \
$data \
$decay \
$feature_type \
$lamda \
$leave_k \
$log \
$lr \
$model \
$num_epoch \
$num_layers \
$num_samples \
$save_params \
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
echo "end"
