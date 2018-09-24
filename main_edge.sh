#!/bin/bash
#SBATCH -o val1.%j.txt
#SBATCH -e val1.%j.err
#SBATCH -t 05:01:00
#SBATCH --mem 3999
#SBATCH -J edge_master
#SBATCH -p shared
#SBATCH --account=ysinger_group

batch_sizes=(100)
bifurcations=(0 0.01)
crossval_ks=(1)
datas=('linqs_cora')
decays=(0)
feature_types=('all')
lamdas=(-14 -12 -10 -8 -6 -4 -2 0 2)
leave_ks=(1)
log='DEBUG'
lrs=(0.1)
models=('att')
num_epochs=(1000)
num_layerss=(10 20 30 40 50 60 70 80 90 100)
num_sampless=(100)
save_params=1
split_seeds=(0 1 2 3 4 5 6 7 8 9)
unlabel_probs=(0.99)
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
