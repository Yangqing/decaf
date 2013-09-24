#!/bin/bash
LAUNCH_ARGS="--port=11237 --launch=slurm --sbatch_args=--partition=vision --sbatch_args=--cpus-per-task=8 --num_clients=10"

python mr_compute_imagenet_features.py \
    --input=synsets.txt \
    --input_folder=/u/vis/x1/common/ILSVRC-2012/train \
    --output_folder=/u/vis/x1/jiayq/ILSVRC2012/jeffnet \
    --output=ILSVRC-jeffnet-log.txt \
    $LAUNCH_ARGS
