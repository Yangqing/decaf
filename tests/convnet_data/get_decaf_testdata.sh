#!/bin/bash
# This script is used together with the cuda-convnet code to generate convolution,
# pooling, and fully connected layers so we can compare agains the decaf
# inplementation.
#
# To generate the decaf test data, simply run the script. 

# MODIFY YOUR PARAMETERS HERE
DATA_DIR='/vis-common/data/CIFAR/cifar-10-py-colmajor/'
CUDA_FOLDER=$HOME/codes/cuda-convnet

# DO NOT CHANGE THINGS BELOW

TEST_FOLDER=$PWD
cd $CUDA_FOLDER

RESULTS_DIR='./results'

if [ -e "$RESULTS_DIR" ]; then
	echo "results dir exists: $RESULTS_DIR"
	echo "exiting"
	exit 1
fi
mkdir -p ${RESULTS_DIR}

LAYER_DEF_FILE='./example-layers/layers-80sec.cfg'
LAYER_PARAM_FILE='./example-layers/layer-params-80sec.cfg'
TEST_RANGE='5'
TRAIN_RANGE='1-4'
GPUID='1'

python \
	convnet.py \
	--data-path=$DATA_DIR \
	--save-path=${RESULTS_DIR} \
	--test-range=$TEST_RANGE \
	--train-range=$TRAIN_RANGE \
	--layer-def=$LAYER_DEF_FILE \
	--layer-params=$LAYER_PARAM_FILE \
	--data-provider=cifar \
	--epochs=12 \
	--gpu=$GPUID \
	--test-freq=48

RESULTS_SUBDIR=`ls -1 $RESULTS_DIR`
RESULTS_FILE=`ls -1 $RESULTS_DIR/$RESULTS_SUBDIR`


for LAYER in data conv1 pool1 pool1_neuron conv2 conv2_neuron pool2 conv3 conv3_neuron pool3 fc64 fc64_neuron fc10 probs
do
    echo "Outputting Layer $LAYER"
    python \
        shownet.py \
        -f $RESULTS_DIR/$RESULTS_SUBDIR \
        --write-features=$LAYER \
        --feature-path=$RESULTS_DIR/80sec-$LAYER \
	    --test-range=$TEST_RANGE \
	    --train-range=$TRAIN_RANGE
done

echo "Dumping layer from $RESULTS_DIR/$RESULTS_SUBDIR/$RESULTS_FILE"
# Now, let's create a layer just
echo "
import cPickle as pickle
temp = pickle.load(open('$RESULTS_DIR/$RESULTS_SUBDIR/$RESULTS_FILE'))
pickle.dump(temp['model_state']['layers'], open('$RESULTS_DIR/layers.pickle', 'w'))
exit()
" | python

# Now copy the data
echo "Copying Data"
cp -r $RESULTS_DIR/* $TEST_FOLDER
echo "Done."
