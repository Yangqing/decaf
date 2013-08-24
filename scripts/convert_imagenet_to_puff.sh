#!/bin/bash

uname -a
mkdir -p /tscratch/tmp/jiayq/

python convert_imagenet_to_puff.py \
    --root=/u/vis/x1/common/ILSVRC-2010/train \
    --output=/tscratch/tmp/jiayq/ilsvrc10-images \
    --output_label=/tscratch/tmp/jiayq/ilsvrc10-labels

echo 'Done'
