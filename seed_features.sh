#!/bin/bash
#SBATCH -o val1.%j.txt
#SBATCH -e val1.%j.err
#SBATCH -t 05:01:00
#SBATCH --mem 1000
#SBATCH -J crossval1
#SBATCH -p shared
#SBATCH --reservation=koshiba


for source in $(seq 0 2484)
do
  for seed in $(seq 0 9)
  do
    echo $source
    ./seed_features_helper.sh \
    'cora' \
    $seed \
    $source
  done
done
echo "cora"

for source in $(seq 0 2109)
do
  for seed in ${seq 0 9}
  do
    echo $source
    ./seed_features_helper.sh \
    'citeseer' \
    $seed \
    $source \
  done
done
echo "citeseer"

for source in $(seq 0 19716)
do
  for seed in ${seq 0 9}
  do
    echo $source
    ./seed_features_helper.sh \
    'pubmed' \
    $seed \
    $source \
  done
done
echo "pubmed"
