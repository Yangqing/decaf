#!/bin/bash

python flask_main.py \
    --net_file=/Users/jiayq/Research/models/imagenet.jeffnet.epoch90 \
    --meta_file=/Users/jiayq/Research/models/imagenet.jeffnet.meta \
    --bet_file=/Users/jiayq/Research/models/imagenet.jeffnet.bet.pickle \
    --upload_folder=./
