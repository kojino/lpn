#!/bin/bash
#SBATCH -o val1.%j.txt
#SBATCH -e val1.%j.err
#SBATCH -t 05:01:00
#SBATCH --mem 3999
#SBATCH -J att_master
#SBATCH -p shared
#SBATCH --reservation=koshiba

change_bs=(0)
clamps=(3)
crossval_ks=(5)
datas=('imdb')
has_featuress=(0)
leave_ks=(1)
logistics=(0)
losss=('log')
loss_class_masss=(1)
lrs=(0.01)
models=('att')
num_epochs=(200)
num_iters=(10)
num_sampless=(100)
parameter_seeds=(0)
# regularizes=(-10 -8 -7 -6 -4)
regularizes=(-30 -18 -16 -14 -12 -10 -8 -6 -4 -2 0 2 4)
regularize_types=('l2')
split_seeds=(0 1 2 3 4 5 6 7 8 9)
asymmetrics=(1)
unlabel_probs=(0.95)
weight_normalizations=('softmax')
confidences=(0)
datatypes=('flip')
profiles=(0)
sparse_edgess=(0)
regularize_weights=(-14)
clamp_scales=(0.01)


for data in ${datas[@]}
do
for crossval_k in ${crossval_ks[@]}
do
for model in ${models[@]}
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
for num_iter in ${num_iters[@]}
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
