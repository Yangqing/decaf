"""A code to perform logistic regression."""
import cPickle as pickle
from decaf import base
from decaf.layers import core_layers
from decaf.layers import regularization
from decaf.layers import fillers
from decaf.layers.data import mnist
from decaf.opt import core_solvers
from decaf.util import visualize
import numpy as np

ROOT_FOLDER='/u/vis/x1/common/mnist'
NUM_NEURONS = 100
NUM_CLASS = 10

def main():
    ######################################
    # First, let's create the decaf layer.
    ######################################
    decaf_net = base.Net()
    # add data layer
    dataset = mnist.MNISTDataLayer(
        name='mnist', rootfolder=ROOT_FOLDER, is_training=True)
    decaf_net.add_layer(dataset, provides=['image', 'label'])
    # add first inner production layer
    ip_layer = core_layers.InnerProductLayer(
        name='ip1', num_output=NUM_NEURONS,
        #reg=regularization.L2Regularizer(weight=0.01),
        filler=fillers.GaussianRandFiller(std=0.1),
        bias_filler=fillers.ConstantFiller(value=0.1))
    decaf_net.add_layer(ip_layer, needs=['image'], provides=['ip1-out'])
    # add ReLU Layer
    relu_layer = core_layers.ReLULayer(name='relu1')
    decaf_net.add_layer(relu_layer, needs=['ip1-out'], provides=['relu1-out'])
    # add second inner production layer
    ip2_layer = core_layers.InnerProductLayer(
        name='ip2', num_output=NUM_CLASS,
        #reg=regularization.L2Regularizer(weight=0.01),
        filler=fillers.GaussianRandFiller(std=0.3))
    decaf_net.add_layer(ip2_layer, needs=['relu1-out'], provides=['prediction'])
    # add loss layer
    loss_layer = core_layers.MultinomialLogisticLossLayer(
        name='loss')
    decaf_net.add_layer(loss_layer, needs=['prediction', 'label'])
    # finish.
    decaf_net.finish()
    ####################################
    # Decaf layer finished construction!
    ####################################
    #raise RuntimeError
    # now, try to solve it
    solver = core_solvers.LBFGSSolver(
        lbfgs_args={'iprint': 1})
    solver.solve(decaf_net)
    # Now let's peek at the accuracy
    accuracy = (decaf_net.blobs['prediction'].data().argmax(1) == \
                decaf_net.blobs['label'].data()).sum() / \
            float(decaf_net.blobs['label'].data().size)
    print 'Training accuracy:', accuracy
    visualize.draw_net_to_file(decaf_net, 'mnist.png')
    decaf_net.save('mnist_2layers.decafnet')

    ##############################################
    # Now, let's try if we are able to load a net.
    ##############################################
    prediction_net = base.Net.load('mnist_2layers.decafnet')
    visualize.draw_net_to_file(prediction_net, 'mnist_test.png')
    dataset_test = mnist.MNISTDataLayer(
        name='mnist', rootfolder=ROOT_FOLDER, is_training=False)
    test_image = base.Blob()
    test_label = base.Blob()
    dataset_test.forward([], [test_image, test_label])
    pred = prediction_net.predict(image=test_image)['prediction']
    accuracy = (pred.argmax(1) == test_label.data()).sum() / float(test_label.data().size)
    print 'Testing accuracy:', accuracy
    print 'Done.'

if __name__ == '__main__':
    main()
