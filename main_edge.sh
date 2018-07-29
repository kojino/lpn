#!/bin/bash
#SBATCH -o val1.%j.txt
#SBATCH -e val1.%j.err
#SBATCH -t 06:50:50
#SBATCH --mem 3999
#SBATCH -J edge_master
#SBATCH -p shared

change_bs=(0)
clamps=(1)
crossval_ks=(1)
datas=('cora')
# datas=('pubmed')
has_featuress=(0)
leave_ks=(1)
logistics=(0)
losss=('log')
loss_class_masss=(1)
lrs=(0.01)
models=('edge')
num_epochs=(3000)
num_iters=(20)
num_sampless=(100)
parameter_seeds=(0)
regularizes=(-30)
regularize_types=('l2')
split_seeds=(0 1 2 3 4 5 6 7 8 9)
asymmetrics=(1)
unlabel_probs=(0.99)
weight_normalizations=('softmax')
confidences=(30)
datatypes=('linqs')
profiles=(0)
sparse_edgess=(0)
regularize_weights=(-14)
clamp_scales=(0.01)

for num_iter in ${num_iters[@]}
do
for crossval_k in ${crossval_ks[@]}
do
for model in ${models[@]}
do
for data in ${datas[@]}
do
for asymmetric in ${asymmetrics[@]}
do
for unlabel_prob in ${unlabel_probs[@]}
do
for leave_k in ${leave_ks[@]}
do
for lr in ${lrs[@]}
do
for clamp_scale in ${clamp_scales[@]}
do
for split_seed in ${split_seeds[@]}
do
for parameter_seed in ${parameter_seeds[@]}
do
for regularize_type in ${regularize_types[@]}
do
for regularize in ${regularizes[@]}
do

for has_features in ${has_featuress[@]}
do
for change_b in ${change_bs[@]}
do
for clamp in ${clamps[@]}
do
for num_epoch in ${num_epochs[@]}
do
for num_samples in ${num_sampless[@]}
do
for logistic in ${logistics[@]}
do
for loss_class_mass in ${loss_class_masss[@]}
do
for weight_normalization in ${weight_normalizations[@]}
do
for confidence in ${confidences[@]}
do
for datatype in ${datatypes[@]}
do
for profile in ${profiles[@]}
do
for sparse_edges in ${sparse_edgess[@]}
do
for regularize_weight in ${regularize_weights[@]}
do
./main_helper.sh \
$change_b \
$clamp \
$crossval_k \
$data \
$has_features \
$leave_k \
$logistic \
$loss_class_mass \
$lr \
$model \
$num_epoch \
$num_iter \
$num_samples \
$parameter_seed \
$regularize \
$regularize_type \
$split_seed \
$asymmetric \
$unlabel_prob \
$weight_normalization \
$confidence \
$datatype \
$profile \
$sparse_edges \
$regularize_weight \
$clamp_scale

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
done
done
done
done
done
done
done
done
echo "end"
