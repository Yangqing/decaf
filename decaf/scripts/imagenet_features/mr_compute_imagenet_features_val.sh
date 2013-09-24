LAUNCH_ARGS="--port=11237 --launch=slurm --sbatch_args=--partition=vision --sbatch_args=--cpus-per-task=8 --num_clients=10"

python mr_compute_imagenet_features.py \
    --input=val.txt \
    --single \
    --input_folder=/u/vis/x1/common/ILSVRC-2012/val \
    --output_folder=/u/vis/x1/jiayq/ILSVRC2012/val \
    --output=ILSVRC-jeffnet-val-log.txt \
    $LAUNCH_ARGS
