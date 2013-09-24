LAUNCH_ARGS="--port=11237 --launch=slurm --sbatch_args=--partition=vision --sbatch_args=--cpus-per-task=8 --num_clients=20"

python mr_compute_imagenet_features.py \
    --input=test.txt \
    --single \
    --input_folder=/u/vis/x1/common/ILSVRC-2012/test \
    --output_folder=/u/vis/x1/jiayq/ILSVRC2012/test \
    --output=ILSVRC-jeffnet-test-log.txt \
    $LAUNCH_ARGS
