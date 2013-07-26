from decaf.layers.data import mnist
from decaf.wraps import logistic_regression
from decaf.util import blasdot
import numpy as np

ROOT_FOLDER='/u/vis/x1/common/mnist'

# let's load the mnist dataset.
dataset = mnist.MNISTDataLayer(
    name='mnist', rootfolder=ROOT_FOLDER, is_training=True)
train_data = dataset._data
train_label = dataset._label

dataset = mnist.MNISTDataLayer(
    name='mnist', rootfolder=ROOT_FOLDER, is_training=False)
test_data = dataset._data
test_label = dataset._label

weight, bias = logistic_regression.logistic_regression(
    train_data, train_label, reg_weight=0.01)

train_score = blasdot.dot(
    train_data.reshape(train_data.shape[0], np.prod(train_data.shape[1:])),
    weight)
train_score += bias
train_pred = train_score.argmax(axis=1)
print 'Train Accuracy:',
print (train_pred == train_label).sum() / float(train_label.shape[0])

test_score = blasdot.dot(
    test_data.reshape(test_data.shape[0],np.prod(test_data.shape[1:])),
    weight)
test_score += bias
test_pred = test_score.argmax(axis=1)
print 'Test Accuracy:',
print (test_pred == test_label).sum() / float(test_label.shape[0])
