#!/bin/bash
#SBATCH --job-name=flask_demo
#SBATCH --partition=vision
#SBATCH --cpus-per-task=1
#SBATCH --mem=4000
#SBATCH --nodelist=orange5

python flask_main.py \
    --net_file=../../scripts/imagenet.jeffnet.epoch72 \
    --meta_file=../../scripts/imagenet.jeffnet.meta
