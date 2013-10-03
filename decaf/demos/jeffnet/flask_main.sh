#!/bin/bash
#SBATCH --job-name=flask_demo
#SBATCH --partition=vision
#SBATCH --exclusive
#SBATCH --cpus-per-task=12
#SBATCH --mem=8000
#SBATCH --nodelist=orange6

export MKL_NUM_THREADS=24
export OMP_NUM_THREADS=12
python flask_main.py \
    --net_file=/u/vis/common/deeplearning/models/imagenet.jeffnet.epoch90 \
    --meta_file=/u/vis/common/deeplearning/models/imagenet.jeffnet.meta \
    --bet_file=/u/vis/common/deeplearning/models/imagenet.jeffnet.bet.pickle
