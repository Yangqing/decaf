#!/bin/bash

python flask_main.py \
    --net_file=/Users/jiayq/Research/models/imagenet.jeffnet.epoch90 \
    --meta_file=/Users/jiayq/Research/models/imagenet.jeffnet.meta \
    --upload_folder=./
