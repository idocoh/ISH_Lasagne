from __future__ import print_function
import time
import os
import sys
import platform

import numpy as np
import theano.sandbox.cuda
from PIL import Image
from lasagne import layers
from lasagne.nonlinearities import tanh
from lasagne.updates import nesterov_momentum
import cPickle as pickle

from featursForSvm import run_svm
from logistic_sgd import load_data
from nolearn.lasagne import BatchIterator
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import PrintLayerInfo
from nolearn.lasagne import TrainSplit
from shape import ReshapeLayer

FILE_SEPARATOR = "/"
counter = 0

LOAD_CAE_PATH = None



def load2d(num_labels, batch_index=1, outputFile=None, input_width=320, input_height=160, end_index=16351, MULTI_POSITIVES=20,
           dropout_percent=0.1, data_set='ISH.pkl.gz', toShuffleInput = False, withZeroMeaning = False, TRAIN_PRECENT=0.8,
           steps=[5000, 10000, 15000, 16352], image_width=320, image_height=160):
    print ('loading data...')

    data_sets = load_data(data_set, batch_index=batch_index, withSVM=0, toShuffleInput=toShuffleInput,
                                               withZeroMeaning=withZeroMeaning, end_index=end_index,
                                               MULTI_POSITIVES=MULTI_POSITIVES, dropout_percent=dropout_percent,
                                               labelset=num_labels, TRAIN_DATA_PRECENT=TRAIN_PRECENT,
                                            steps=steps, image_width=image_width, image_height=image_height)

    train_set_x, train_set_y = data_sets[0]
#     valid_set_x, valid_set_y = data_sets[1]
    test_set_x, test_set_y = data_sets[2]
    print(train_set_x.shape[0], ' samples loaded')
    return (train_set_x, train_set_y, test_set_x, test_set_y)


class Unpool2DLayer(layers.Layer):
    """
    This layer performs unpooling over the last two dimensions
    of a 4D tensor.
    """
    def __init__(self, incoming, ds, **kwargs):

        super(Unpool2DLayer, self).__init__(incoming, **kwargs)

        if (isinstance(ds, int)):
            raise ValueError('ds must have len == 2')
        else:
            ds = tuple(ds)
            if len(ds) != 2:
                raise ValueError('ds must have len == 2')
            if ds[0] != ds[1]:
                raise ValueError('ds should be symmetric (I am lazy)')
            self.ds = ds

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)

        output_shape[2] = input_shape[2] * self.ds[0]
        output_shape[3] = input_shape[3] * self.ds[1]

        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        ds = self.ds
        input_shape = input.shape
        output_shape = self.get_output_shape_for(input_shape)
        return input.repeat(2, axis=2).repeat(2, axis=3)


### when we load the batches to input to the neural network, we randomly / flip
### rotate the images, to artificially increase the size of the training set
class FlipBatchIterator(BatchIterator):
    def transform(self, X1, X2):
        X1b, X2b = super(FlipBatchIterator, self).transform(X1, X2)
        X2b = X2b.reshape(X1b.shape)

        bs = X1b.shape[0]
        h_indices = np.random.choice(bs, bs / 2, replace=False)  # horizontal flip
        v_indices = np.random.choice(bs, bs / 2, replace=False)  # vertical flip

        ###  uncomment these lines if you want to include rotations (images must be square)  ###
        #r_indices = np.random.choice(bs, bs / 2, replace=False) # 90 degree rotation
        for X in (X1b, X2b):
            X[h_indices] = X[h_indices, :, :, ::-1]
            X[v_indices] = X[v_indices, :, ::-1, :]
            #X[r_indices] = np.swapaxes(X[r_indices, :, :, :], 2, 3)
        shape = X2b.shape
        X2b = X2b.reshape((shape[0], -1))

        return X1b, X2b


def run(loadedData=None, learning_rate=0.04, update_momentum=0.9, update_rho=None, epochs=15, number_pooling_layers=2,
        input_width=320, input_height=160, train_valid_split=0.2, multiple_positives=20, flip_batch=True,
        dropout_percent=0.1, amount_train=16351, activation=None, last_layer_activation=None, batch_size=32,
        layers_size=[5, 10, 20, 40], shuffle_input=False, zero_meaning=False, filters_type=3,
        categories=15, svm_negative_amount=800, folder_name="default", number_conv_layers=4):

    global counter
    folder_path = "results_dae"+FILE_SEPARATOR + folder_name + FILE_SEPARATOR + "run_" + str(counter) + FILE_SEPARATOR
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    All_Results_FIle = "results_dae"+FILE_SEPARATOR + "all_results.txt"
    PARAMS_FILE_NAME = folder_path + "parameters.txt"
    # HIDDEN_LAYER_OUTPUT_FILE_NAME = folder_path + "hiddenLayerOutput.pkl.gz"
    # FIG_FILE_NAME = folder_path + "fig"
    # PICKLES_NET_FILE_NAME = folder_path + "picklesNN.pkl.gz"
    # SVM_FILE_NAME = folder_path + "svmData.txt"
    # LOG_FILE_NAME = folder_path + "message.log"



    #     old_stdout = sys.stdout
    #     print "less",LOG_FILE_NAME
    # log_file = False  #open(LOG_FILE_NAME, "w")
    #     sys.stdout = log_file

    counter += 1
    output_file = open(PARAMS_FILE_NAME, "w")
    results_file = open(All_Results_FIle, "a")

    def create_cae():

        def create_cae_4pool_4conv(input_height, input_width):

            cnn = NeuralNet(layers=[
                ('input', layers.InputLayer),
                ('conv1', layers.Conv2DLayer),
                ('conv11', layers.Conv2DLayer),
                ('conv12', layers.Conv2DLayer),
                ('conv13', layers.Conv2DLayer),

                ('pool1', layers.MaxPool2DLayer),

                ('conv2', layers.Conv2DLayer),
                ('conv21', layers.Conv2DLayer),
                ('conv22', layers.Conv2DLayer),
                ('conv23', layers.Conv2DLayer),

                ('pool2', layers.MaxPool2DLayer),

                ('conv3', layers.Conv2DLayer),
                ('conv31', layers.Conv2DLayer),
                ('conv32', layers.Conv2DLayer),
                ('conv33', layers.Conv2DLayer),

                ('pool3', layers.MaxPool2DLayer),

                ('conv4', layers.Conv2DLayer),
                ('conv41', layers.Conv2DLayer),
                ('conv42', layers.Conv2DLayer),
                ('conv43', layers.Conv2DLayer),

                ('pool4', layers.MaxPool2DLayer),

                ('conv5', layers.Conv2DLayer),
                ('conv51', layers.Conv2DLayer),
                ('conv52', layers.Conv2DLayer),
                ('conv53', layers.Conv2DLayer),

                ('unpool1', Unpool2DLayer),

                ('conv6', layers.Conv2DLayer),
                ('conv61', layers.Conv2DLayer),
                ('conv62', layers.Conv2DLayer),
                ('conv63', layers.Conv2DLayer),

                ('unpool2', Unpool2DLayer),

                ('conv7', layers.Conv2DLayer),
                ('conv71', layers.Conv2DLayer),
                ('conv72', layers.Conv2DLayer),
                ('conv73', layers.Conv2DLayer),

                ('unpool3', Unpool2DLayer),

                ('conv8', layers.Conv2DLayer),
                ('conv81', layers.Conv2DLayer),
                ('conv82', layers.Conv2DLayer),
                ('conv83', layers.Conv2DLayer),

                ('unpool4', Unpool2DLayer),

                ('conv9', layers.Conv2DLayer),
                ('conv91', layers.Conv2DLayer),
                ('conv92', layers.Conv2DLayer),
                ('conv93', layers.Conv2DLayer),

                ('conv10', layers.Conv2DLayer),

                ('output_layer', ReshapeLayer),
            ],

                input_shape=(None, 1, input_width, input_height),
                # Layer current size - 1x320x160

                conv1_num_filters=layers_size[0], conv1_filter_size=filter_1, conv1_nonlinearity=activation,
                # conv1_border_mode="same",
                conv1_pad="same",
                conv11_num_filters=layers_size[0], conv11_filter_size=filter_1, conv11_nonlinearity=activation,
                # conv11_border_mode="same",
                conv11_pad="same",
                conv12_num_filters=layers_size[0], conv12_filter_size=filter_1, conv12_nonlinearity=activation,
                # conv12_border_mode="same",
                conv12_pad="same",
                conv13_num_filters=layers_size[0], conv13_filter_size=filter_1, conv13_nonlinearity=activation,
                # conv13_border_mode="same",
                conv13_pad="same",

                pool1_pool_size=(2, 2),

                conv2_num_filters=layers_size[1], conv2_filter_size=filter_2, conv2_nonlinearity=activation,
                # conv2_border_mode="same",
                conv2_pad="same",
                conv21_num_filters=layers_size[1], conv21_filter_size=filter_2, conv21_nonlinearity=activation,
                # conv21_border_mode="same",
                conv21_pad="same",
                conv22_num_filters=layers_size[1], conv22_filter_size=filter_2, conv22_nonlinearity=activation,
                # conv22_border_mode="same",
                conv22_pad="same",
                conv23_num_filters=layers_size[1], conv23_filter_size=filter_2, conv23_nonlinearity=activation,
                # conv23_border_mode="same",
                conv23_pad="same",

                pool2_pool_size=(2, 2),

                conv3_num_filters=layers_size[2], conv3_filter_size=filter_3, conv3_nonlinearity=activation,
                # conv3_border_mode="same",
                conv3_pad="same",
                conv31_num_filters=layers_size[2], conv31_filter_size=filter_3, conv31_nonlinearity=activation,
                # conv31_border_mode="same",
                conv31_pad="same",
                conv32_num_filters=layers_size[2], conv32_filter_size=filter_3, conv32_nonlinearity=activation,
                # conv32_border_mode="same",
                conv32_pad="same",
                conv33_num_filters=layers_size[2], conv33_filter_size=filter_3, conv33_nonlinearity=activation,
                # conv33_border_mode="same",
                conv33_pad="same",

                pool3_pool_size=(2, 2),

                conv4_num_filters=layers_size[3], conv4_filter_size=filter_4, conv4_nonlinearity=activation,
                # conv4_border_mode="same",
                conv4_pad="same",
                conv41_num_filters=layers_size[3], conv41_filter_size=filter_4, conv41_nonlinearity=activation,
                # conv41_border_mode="same",
                conv41_pad="same",
                conv42_num_filters=layers_size[3], conv42_filter_size=filter_4, conv42_nonlinearity=activation,
                # conv42_border_mode="same",
                conv42_pad="same",
                conv43_num_filters=layers_size[3], conv43_filter_size=filter_4, conv43_nonlinearity=activation,
                # conv43_border_mode="same",
                conv43_pad="same",

                pool4_pool_size=(2, 2),

                conv5_num_filters=layers_size[4], conv5_filter_size=filter_5, conv5_nonlinearity=activation,
                # conv5_border_mode="same",
                conv5_pad="same",
                conv51_num_filters=layers_size[4], conv51_filter_size=filter_5, conv51_nonlinearity=activation,
                # conv51_border_mode="same",
                conv51_pad="same",
                conv52_num_filters=layers_size[4], conv52_filter_size=filter_5, conv52_nonlinearity=activation,
                # conv52_border_mode="same",
                conv52_pad="same",
                conv53_num_filters=1, conv53_filter_size=filter_5, conv53_nonlinearity=activation,
                # conv53_border_mode="same",
                conv53_pad="same",

                unpool1_ds=(2, 2),

                conv6_num_filters=layers_size[5], conv6_filter_size=filter_6, conv6_nonlinearity=activation,
                # conv6_border_mode="same",
                conv6_pad="same",
                conv61_num_filters=layers_size[5], conv61_filter_size=filter_6, conv61_nonlinearity=activation,
                # conv61_border_mode="same",
                conv61_pad="same",
                conv62_num_filters=layers_size[5], conv62_filter_size=filter_6, conv62_nonlinearity=activation,
                # conv62_border_mode="same",
                conv62_pad="same",
                conv63_num_filters=layers_size[5], conv63_filter_size=filter_6, conv63_nonlinearity=activation,
                # conv63_border_mode="same",
                conv63_pad="same",

                unpool2_ds=(2, 2),

                conv7_num_filters=layers_size[6], conv7_filter_size=filter_7, conv7_nonlinearity=activation,
                # conv7_border_mode="same",
                conv7_pad="same",
                conv71_num_filters=layers_size[6], conv71_filter_size=filter_7, conv71_nonlinearity=activation,
                # conv71_border_mode="same",
                conv71_pad="same",
                conv72_num_filters=layers_size[6], conv72_filter_size=filter_7, conv72_nonlinearity=activation,
                # conv72_border_mode="same",
                conv72_pad="same",
                conv73_num_filters=layers_size[6], conv73_filter_size=filter_7, conv73_nonlinearity=activation,
                # conv73_border_mode="same",
                conv73_pad="same",

                unpool3_ds=(2, 2),

                conv8_num_filters=layers_size[7], conv8_filter_size=filter_8, conv8_nonlinearity=activation,
                # conv8_border_mode="same",
                conv8_pad="same",
                conv81_num_filters=layers_size[7], conv81_filter_size=filter_8, conv81_nonlinearity=activation,
                # conv81_border_mode="same",
                conv81_pad="same",
                conv82_num_filters=layers_size[7], conv82_filter_size=filter_8, conv82_nonlinearity=activation,
                # conv82_border_mode="same",
                conv82_pad="same",
                conv83_num_filters=layers_size[7], conv83_filter_size=filter_8, conv83_nonlinearity=activation,
                # conv83_border_mode="same",
                conv83_pad="same",

                unpool4_ds=(2, 2),

                conv9_num_filters=layers_size[8], conv9_filter_size=filter_9, conv9_nonlinearity=activation,
                # conv9_border_mode="same",
                conv9_pad="same",
                conv91_num_filters=layers_size[8], conv91_filter_size=filter_9, conv91_nonlinearity=activation,
                # conv91_border_mode="same",
                conv91_pad="same",
                conv92_num_filters=layers_size[8], conv92_filter_size=filter_9, conv92_nonlinearity=activation,
                # conv92_border_mode="same",
                conv92_pad="same",
                conv93_num_filters=layers_size[8], conv93_filter_size=filter_9, conv93_nonlinearity=activation,
                # conv93_border_mode="same",
                conv93_pad="same",

                conv10_num_filters=1, conv10_filter_size=filter_10, conv10_nonlinearity=last_layer_activation,
                # conv10_border_mode="same",
                conv10_pad="same",

                output_layer_shape=(([0], -1)),

                update_learning_rate=learning_rate,
                update_momentum=update_momentum,
                update=nesterov_momentum,
                train_split=TrainSplit(eval_size=train_valid_split),
                batch_iterator_train=FlipBatchIterator(batch_size=batch_size) if flip_batch else BatchIterator(
                    batch_size=batch_size),
                regression=True,
                max_epochs=epochs,
                verbose=1,
                hiddenLayer_to_output=24)  # 19)

            return cnn

        def create_cae_3pool_4conv(input_height, input_width):

            cnn = NeuralNet(layers=[
                ('input', layers.InputLayer),
                ('conv1', layers.Conv2DLayer),
                ('conv11', layers.Conv2DLayer),
                ('conv12', layers.Conv2DLayer),
                ('conv13', layers.Conv2DLayer),

                ('pool1', layers.MaxPool2DLayer),

                ('conv2', layers.Conv2DLayer),
                ('conv21', layers.Conv2DLayer),
                ('conv22', layers.Conv2DLayer),
                ('conv23', layers.Conv2DLayer),

                ('pool2', layers.MaxPool2DLayer),

                ('conv3', layers.Conv2DLayer),
                ('conv31', layers.Conv2DLayer),
                ('conv32', layers.Conv2DLayer),
                ('conv33', layers.Conv2DLayer),

                ('pool3', layers.MaxPool2DLayer),

                ('conv4', layers.Conv2DLayer),
                ('conv41', layers.Conv2DLayer),
                ('conv42', layers.Conv2DLayer),
                ('conv43', layers.Conv2DLayer),

                ('unpool1', Unpool2DLayer),

                ('conv5', layers.Conv2DLayer),
                ('conv51', layers.Conv2DLayer),
                ('conv52', layers.Conv2DLayer),
                ('conv53', layers.Conv2DLayer),

                ('unpool2', Unpool2DLayer),

                ('conv6', layers.Conv2DLayer),
                ('conv61', layers.Conv2DLayer),
                ('conv62', layers.Conv2DLayer),
                ('conv63', layers.Conv2DLayer),

                ('unpool3', Unpool2DLayer),

                ('conv7', layers.Conv2DLayer),
                ('conv71', layers.Conv2DLayer),
                ('conv72', layers.Conv2DLayer),
                ('conv73', layers.Conv2DLayer),

                ('conv8', layers.Conv2DLayer),

                ('output_layer', ReshapeLayer),
            ],

                input_shape=(None, 1, input_width, input_height),
                # Layer current size - 1x320x160

                conv1_num_filters=layers_size[0], conv1_filter_size=filter_1, conv1_nonlinearity=activation,
                # conv1_border_mode="same",
                conv1_pad="same",
                conv11_num_filters=layers_size[0], conv11_filter_size=filter_1, conv11_nonlinearity=activation,
                # conv11_border_mode="same",
                conv11_pad="same",
                conv12_num_filters=layers_size[0], conv12_filter_size=filter_1, conv12_nonlinearity=activation,
                # conv12_border_mode="same",
                conv12_pad="same",
                conv13_num_filters=layers_size[0], conv13_filter_size=filter_1, conv13_nonlinearity=activation,
                # conv13_border_mode="same",
                conv13_pad="same",

                pool1_pool_size=(2, 2),

                conv2_num_filters=layers_size[1], conv2_filter_size=filter_2, conv2_nonlinearity=activation,
                # conv2_border_mode="same",
                conv2_pad="same",
                conv21_num_filters=layers_size[1], conv21_filter_size=filter_2, conv21_nonlinearity=activation,
                # conv21_border_mode="same",
                conv21_pad="same",
                conv22_num_filters=layers_size[1], conv22_filter_size=filter_2, conv22_nonlinearity=activation,
                # conv22_border_mode="same",
                conv22_pad="same",
                conv23_num_filters=layers_size[1], conv23_filter_size=filter_2, conv23_nonlinearity=activation,
                # conv23_border_mode="same",
                conv23_pad="same",

                pool2_pool_size=(2, 2),

                conv3_num_filters=layers_size[2], conv3_filter_size=filter_3, conv3_nonlinearity=activation,
                # conv3_border_mode="same",
                conv3_pad="same",
                conv31_num_filters=layers_size[2], conv31_filter_size=filter_3, conv31_nonlinearity=activation,
                # conv31_border_mode="same",
                conv31_pad="same",
                conv32_num_filters=layers_size[2], conv32_filter_size=filter_3, conv32_nonlinearity=activation,
                # conv32_border_mode="same",
                conv32_pad="same",
                conv33_num_filters=layers_size[2], conv33_filter_size=filter_3, conv33_nonlinearity=activation,
                # conv33_border_mode="same",
                conv33_pad="same",

                pool3_pool_size=(2, 2),

                conv4_num_filters=layers_size[3], conv4_filter_size=filter_4, conv4_nonlinearity=activation,
                # conv4_border_mode="same",
                conv4_pad="same",
                conv41_num_filters=layers_size[3], conv41_filter_size=filter_4, conv41_nonlinearity=activation,
                # conv41_border_mode="same",
                conv41_pad="same",
                conv42_num_filters=layers_size[3], conv42_filter_size=filter_4, conv42_nonlinearity=activation,
                # conv42_border_mode="same",
                conv42_pad="same",
                conv43_num_filters=1, conv43_filter_size=filter_4, conv43_nonlinearity=activation,
                # conv43_border_mode="same",
                conv43_pad="same",

                unpool1_ds=(2, 2),

                conv5_num_filters=layers_size[4], conv5_filter_size=filter_5, conv5_nonlinearity=activation,
                # conv5_border_mode="same",
                conv5_pad="same",
                conv51_num_filters=layers_size[4], conv51_filter_size=filter_5, conv51_nonlinearity=activation,
                # conv51_border_mode="same",
                conv51_pad="same",
                conv52_num_filters=layers_size[4], conv52_filter_size=filter_5, conv52_nonlinearity=activation,
                # conv52_border_mode="same",
                conv52_pad="same",
                conv53_num_filters=layers_size[4], conv53_filter_size=filter_5, conv53_nonlinearity=activation,
                # conv53_border_mode="same",
                conv53_pad="same",

                unpool2_ds=(2, 2),

                conv6_num_filters=layers_size[5], conv6_filter_size=filter_6, conv6_nonlinearity=activation,
                # conv6_border_mode="same",
                conv6_pad="same",
                conv61_num_filters=layers_size[5], conv61_filter_size=filter_6, conv61_nonlinearity=activation,
                # conv61_border_mode="same",
                conv61_pad="same",
                conv62_num_filters=layers_size[5], conv62_filter_size=filter_6, conv62_nonlinearity=activation,
                # conv62_border_mode="same",
                conv62_pad="same",
                conv63_num_filters=layers_size[5], conv63_filter_size=filter_6, conv63_nonlinearity=activation,
                # conv63_border_mode="same",
                conv63_pad="same",

                unpool3_ds=(2, 2),

                conv7_num_filters=layers_size[6], conv7_filter_size=filter_7, conv7_nonlinearity=activation,
                # conv7_border_mode="same",
                conv7_pad="same",
                conv71_num_filters=layers_size[6], conv71_filter_size=filter_7, conv71_nonlinearity=activation,
                # conv71_border_mode="same",
                conv71_pad="same",
                conv72_num_filters=layers_size[6], conv72_filter_size=filter_7, conv72_nonlinearity=activation,
                # conv72_border_mode="same",
                conv72_pad="same",
                conv73_num_filters=layers_size[6], conv73_filter_size=filter_7, conv73_nonlinearity=activation,
                # conv73_border_mode="same",
                conv73_pad="same",

                conv8_num_filters=1, conv8_filter_size=filter_8, conv8_nonlinearity=last_layer_activation,
                # conv8_border_mode="same",
                conv8_pad="same",

                output_layer_shape=(([0], -1)),

                update_learning_rate=learning_rate,
                update_momentum=update_momentum,
                update=nesterov_momentum,
                train_split=TrainSplit(eval_size=train_valid_split),
                batch_iterator_train=FlipBatchIterator(batch_size=batch_size) if flip_batch else BatchIterator(batch_size=batch_size),
                regression=True,
                max_epochs=epochs,
                verbose=1,
                hiddenLayer_to_output=19)#15)#

            return cnn

        def create_cae_2pool_4conv(input_height, input_width):

            cnn = NeuralNet(layers=[
                ('input', layers.InputLayer),
                ('conv1', layers.Conv2DLayer),
                ('conv11', layers.Conv2DLayer),
                ('conv12', layers.Conv2DLayer),
                ('conv13', layers.Conv2DLayer),
                ('pool1', layers.MaxPool2DLayer),
                ('conv2', layers.Conv2DLayer),
                ('conv21', layers.Conv2DLayer),
                ('conv22', layers.Conv2DLayer),
                ('conv23', layers.Conv2DLayer),
                ('pool2', layers.MaxPool2DLayer),
                ('conv3', layers.Conv2DLayer),
                ('conv31', layers.Conv2DLayer),
                ('conv32', layers.Conv2DLayer),
                ('conv33', layers.Conv2DLayer),
                ('unpool1', Unpool2DLayer),
                ('conv4', layers.Conv2DLayer),
                ('conv41', layers.Conv2DLayer),
                ('conv42', layers.Conv2DLayer),
                ('conv43', layers.Conv2DLayer),
                ('unpool2', Unpool2DLayer),
                ('conv5', layers.Conv2DLayer),
                ('conv51', layers.Conv2DLayer),
                ('conv52', layers.Conv2DLayer),
                ('conv53', layers.Conv2DLayer),
                ('conv6', layers.Conv2DLayer),
                ('output_layer', ReshapeLayer),
            ],

                input_shape=(None, 1, input_width, input_height),
                # Layer current size - 1x300x140

                conv1_num_filters=layers_size[0], conv1_filter_size=filter_1, conv1_nonlinearity=activation,
                # conv1_border_mode="same",
                conv1_pad="same",
                conv11_num_filters=layers_size[0], conv11_filter_size=filter_1, conv11_nonlinearity=activation,
                # conv11_border_mode="same",
                conv11_pad="same",
                conv12_num_filters=layers_size[0], conv12_filter_size=filter_1, conv12_nonlinearity=activation,
                # conv12_border_mode="same",
                conv12_pad="same",
                conv13_num_filters=layers_size[0], conv13_filter_size=filter_1, conv13_nonlinearity=activation,
                # conv13_border_mode="same",
                conv13_pad="same",

                pool1_pool_size=(2, 2),

                conv2_num_filters=layers_size[1], conv2_filter_size=filter_2, conv2_nonlinearity=activation,
                # conv2_border_mode="same",
                conv2_pad="same",
                conv21_num_filters=layers_size[1], conv21_filter_size=filter_2, conv21_nonlinearity=activation,
                # conv21_border_mode="same",
                conv21_pad="same",
                conv22_num_filters=layers_size[1], conv22_filter_size=filter_2, conv22_nonlinearity=activation,
                # conv22_border_mode="same",
                conv22_pad="same",
                conv23_num_filters=layers_size[1], conv23_filter_size=filter_2, conv23_nonlinearity=activation,
                # conv23_border_mode="same",
                conv23_pad="same",

                pool2_pool_size=(2, 2),

                conv3_num_filters=layers_size[2], conv3_filter_size=filter_3, conv3_nonlinearity=activation,
                # conv3_border_mode="same",
                conv3_pad="same",
                conv31_num_filters=layers_size[2], conv31_filter_size=filter_3, conv31_nonlinearity=activation,
                # conv31_border_mode="same",
                conv31_pad="same",
                conv32_num_filters=layers_size[2], conv32_filter_size=filter_3, conv32_nonlinearity=activation,
                # conv32_border_mode="same",
                conv32_pad="same",
                conv33_num_filters=1, conv33_filter_size=filter_3, conv33_nonlinearity=activation,
                # conv33_border_mode="same",
                conv33_pad="same",

                unpool1_ds=(2, 2),

                conv4_num_filters=layers_size[3], conv4_filter_size=filter_4, conv4_nonlinearity=activation,
                # conv4_border_mode="same",
                conv4_pad="same",
                conv41_num_filters=layers_size[3], conv41_filter_size=filter_4, conv41_nonlinearity=activation,
                # conv41_border_mode="same",
                conv41_pad="same",
                conv42_num_filters=layers_size[3], conv42_filter_size=filter_4, conv42_nonlinearity=activation,
                # conv42_border_mode="same",
                conv42_pad="same",
                conv43_num_filters=layers_size[3], conv43_filter_size=filter_4, conv43_nonlinearity=activation,
                # conv43_border_mode="same",
                conv43_pad="same",

                unpool2_ds=(2, 2),

                conv5_num_filters=layers_size[4], conv5_filter_size=filter_5, conv5_nonlinearity=activation,
                # conv5_border_mode="same",
                conv5_pad="same",
                conv51_num_filters=layers_size[4], conv51_filter_size=filter_5, conv51_nonlinearity=activation,
                # conv51_border_mode="same",
                conv51_pad="same",
                conv52_num_filters=layers_size[4], conv52_filter_size=filter_5, conv52_nonlinearity=activation,
                # conv52_border_mode="same",
                conv52_pad="same",
                conv53_num_filters=layers_size[4], conv53_filter_size=filter_5, conv53_nonlinearity=activation,
                # conv53_border_mode="same",
                conv53_pad="same",

                conv6_num_filters=1, conv6_filter_size=filter_6, conv6_nonlinearity=last_layer_activation,
                # conv6_border_mode="same",
                conv6_pad="same",

                output_layer_shape=(([0], -1)),

                update_learning_rate=learning_rate,
                update_momentum=update_momentum,
                update=nesterov_momentum,
                train_split=TrainSplit(eval_size=train_valid_split),
                batch_iterator_train=FlipBatchIterator(batch_size=batch_size) if flip_batch else BatchIterator(batch_size=batch_size),
                regression=True,
                max_epochs=epochs,
                verbose=1,
                hiddenLayer_to_output=14)

            return cnn

        def create_cae_4pool_3conv(input_height, input_width):

            cnn = NeuralNet(layers=[
                ('input', layers.InputLayer),
                ('conv1', layers.Conv2DLayer),
                ('conv11', layers.Conv2DLayer),
                ('conv12', layers.Conv2DLayer),

                ('pool1', layers.MaxPool2DLayer),

                ('conv2', layers.Conv2DLayer),
                ('conv21', layers.Conv2DLayer),
                ('conv22', layers.Conv2DLayer),

                ('pool2', layers.MaxPool2DLayer),

                ('conv3', layers.Conv2DLayer),
                ('conv31', layers.Conv2DLayer),
                ('conv32', layers.Conv2DLayer),

                ('pool3', layers.MaxPool2DLayer),

                ('conv4', layers.Conv2DLayer),
                ('conv41', layers.Conv2DLayer),
                ('conv42', layers.Conv2DLayer),

                ('pool4', layers.MaxPool2DLayer),

                ('conv5', layers.Conv2DLayer),
                ('conv51', layers.Conv2DLayer),
                ('conv52', layers.Conv2DLayer),

                ('unpool1', Unpool2DLayer),

                ('conv6', layers.Conv2DLayer),
                ('conv61', layers.Conv2DLayer),
                ('conv62', layers.Conv2DLayer),

                ('unpool2', Unpool2DLayer),

                ('conv7', layers.Conv2DLayer),
                ('conv71', layers.Conv2DLayer),
                ('conv72', layers.Conv2DLayer),

                ('unpool3', Unpool2DLayer),

                ('conv8', layers.Conv2DLayer),
                ('conv81', layers.Conv2DLayer),
                ('conv82', layers.Conv2DLayer),

                ('unpool4', Unpool2DLayer),

                ('conv9', layers.Conv2DLayer),
                ('conv91', layers.Conv2DLayer),
                ('conv92', layers.Conv2DLayer),

                ('conv10', layers.Conv2DLayer),

                ('output_layer', ReshapeLayer),
            ],

                input_shape=(None, 1, input_width, input_height),
                # Layer current size - 1x320x160

                conv1_num_filters=layers_size[0], conv1_filter_size=filter_1, conv1_nonlinearity=activation,
                # conv1_border_mode="same",
                conv1_pad="same",
                conv11_num_filters=layers_size[0], conv11_filter_size=filter_1, conv11_nonlinearity=activation,
                # conv11_border_mode="same",
                conv11_pad="same",
                conv12_num_filters=layers_size[0], conv12_filter_size=filter_1, conv12_nonlinearity=activation,
                # conv12_border_mode="same",
                conv12_pad="same",

                pool1_pool_size=(2, 2),

                conv2_num_filters=layers_size[1], conv2_filter_size=filter_2, conv2_nonlinearity=activation,
                # conv2_border_mode="same",
                conv2_pad="same",
                conv21_num_filters=layers_size[1], conv21_filter_size=filter_2, conv21_nonlinearity=activation,
                # conv21_border_mode="same",
                conv21_pad="same",
                conv22_num_filters=layers_size[1], conv22_filter_size=filter_2, conv22_nonlinearity=activation,
                # conv22_border_mode="same",
                conv22_pad="same",

                pool2_pool_size=(2, 2),

                conv3_num_filters=layers_size[2], conv3_filter_size=filter_3, conv3_nonlinearity=activation,
                # conv3_border_mode="same",
                conv3_pad="same",
                conv31_num_filters=layers_size[2], conv31_filter_size=filter_3, conv31_nonlinearity=activation,
                # conv31_border_mode="same",
                conv31_pad="same",
                conv32_num_filters=layers_size[2], conv32_filter_size=filter_3, conv32_nonlinearity=activation,
                # conv32_border_mode="same",
                conv32_pad="same",

                pool3_pool_size=(2, 2),

                conv4_num_filters=layers_size[3], conv4_filter_size=filter_4, conv4_nonlinearity=activation,
                # conv4_border_mode="same",
                conv4_pad="same",
                conv41_num_filters=layers_size[3], conv41_filter_size=filter_4, conv41_nonlinearity=activation,
                # conv41_border_mode="same",
                conv41_pad="same",
                conv42_num_filters=layers_size[3], conv42_filter_size=filter_4, conv42_nonlinearity=activation,
                # conv42_border_mode="same",
                conv42_pad="same",

                pool4_pool_size=(2, 2),

                conv5_num_filters=layers_size[4], conv5_filter_size=filter_5, conv5_nonlinearity=activation,
                # conv5_border_mode="same",
                conv5_pad="same",
                conv51_num_filters=layers_size[4], conv51_filter_size=filter_5, conv51_nonlinearity=activation,
                # conv51_border_mode="same",
                conv51_pad="same",
                conv52_num_filters=1, conv52_filter_size=filter_5, conv52_nonlinearity=activation,
                # conv52_border_mode="same",
                conv52_pad="same",

                unpool1_ds=(2, 2),

                conv6_num_filters=layers_size[5], conv6_filter_size=filter_6, conv6_nonlinearity=activation,
                # conv6_border_mode="same",
                conv6_pad="same",
                conv61_num_filters=layers_size[5], conv61_filter_size=filter_6, conv61_nonlinearity=activation,
                # conv61_border_mode="same",
                conv61_pad="same",
                conv62_num_filters=layers_size[5], conv62_filter_size=filter_6, conv62_nonlinearity=activation,
                # conv62_border_mode="same",
                conv62_pad="same",

                unpool2_ds=(2, 2),

                conv7_num_filters=layers_size[6], conv7_filter_size=filter_7, conv7_nonlinearity=activation,
                # conv7_border_mode="same",
                conv7_pad="same",
                conv71_num_filters=layers_size[6], conv71_filter_size=filter_7, conv71_nonlinearity=activation,
                # conv71_border_mode="same",
                conv71_pad="same",
                conv72_num_filters=layers_size[6], conv72_filter_size=filter_7, conv72_nonlinearity=activation,
                # conv72_border_mode="same",
                conv72_pad="same",

                unpool3_ds=(2, 2),

                conv8_num_filters=layers_size[7], conv8_filter_size=filter_8, conv8_nonlinearity=activation,
                # conv8_border_mode="same",
                conv8_pad="same",
                conv81_num_filters=layers_size[7], conv81_filter_size=filter_8, conv81_nonlinearity=activation,
                # conv81_border_mode="same",
                conv81_pad="same",
                conv82_num_filters=layers_size[7], conv82_filter_size=filter_8, conv82_nonlinearity=activation,
                # conv82_border_mode="same",
                conv82_pad="same",

                unpool4_ds=(2, 2),

                conv9_num_filters=layers_size[8], conv9_filter_size=filter_9, conv9_nonlinearity=activation,
                # conv9_border_mode="same",
                conv9_pad="same",
                conv91_num_filters=layers_size[8], conv91_filter_size=filter_9, conv91_nonlinearity=activation,
                # conv91_border_mode="same",
                conv91_pad="same",
                conv92_num_filters=layers_size[8], conv92_filter_size=filter_9, conv92_nonlinearity=activation,
                # conv92_border_mode="same",
                conv92_pad="same",

                conv10_num_filters=1, conv10_filter_size=filter_10, conv10_nonlinearity=last_layer_activation,
                # conv10_border_mode="same",
                conv10_pad="same",

                output_layer_shape=(([0], -1)),

                update_learning_rate=learning_rate,
                update_momentum=update_momentum,
                update=nesterov_momentum,
                train_split=TrainSplit(eval_size=train_valid_split),
                batch_iterator_train=FlipBatchIterator(batch_size=batch_size) if flip_batch else BatchIterator(
                    batch_size=batch_size),
                regression=True,
                max_epochs=epochs,
                verbose=1,
                hiddenLayer_to_output=19)

            return cnn

        def create_cae_3pool_3conv(input_height, input_width):

            cnn = NeuralNet(layers=[
                ('input', layers.InputLayer),
                ('conv1', layers.Conv2DLayer),
                ('conv11', layers.Conv2DLayer),
                ('conv12', layers.Conv2DLayer),

                ('pool1', layers.MaxPool2DLayer),

                ('conv2', layers.Conv2DLayer),
                ('conv21', layers.Conv2DLayer),
                ('conv22', layers.Conv2DLayer),

                ('pool2', layers.MaxPool2DLayer),

                ('conv3', layers.Conv2DLayer),
                ('conv31', layers.Conv2DLayer),
                ('conv32', layers.Conv2DLayer),

                ('pool3', layers.MaxPool2DLayer),

                ('conv4', layers.Conv2DLayer),
                ('conv41', layers.Conv2DLayer),
                ('conv42', layers.Conv2DLayer),

                ('unpool1', Unpool2DLayer),

                ('conv5', layers.Conv2DLayer),
                ('conv51', layers.Conv2DLayer),
                ('conv52', layers.Conv2DLayer),

                ('unpool2', Unpool2DLayer),

                ('conv6', layers.Conv2DLayer),
                ('conv61', layers.Conv2DLayer),
                ('conv62', layers.Conv2DLayer),

                ('unpool3', Unpool2DLayer),

                ('conv7', layers.Conv2DLayer),
                ('conv71', layers.Conv2DLayer),
                ('conv72', layers.Conv2DLayer),

                ('conv8', layers.Conv2DLayer),

                ('output_layer', ReshapeLayer),
            ],

                input_shape=(None, 1, input_width, input_height),
                # Layer current size - 1x320x160

                conv1_num_filters=layers_size[0], conv1_filter_size=filter_1, conv1_nonlinearity=activation,
                # conv1_border_mode="same",
                conv1_pad="same",
                conv11_num_filters=layers_size[0], conv11_filter_size=filter_1, conv11_nonlinearity=activation,
                # conv11_border_mode="same",
                conv11_pad="same",
                conv12_num_filters=layers_size[0], conv12_filter_size=filter_1, conv12_nonlinearity=activation,
                # conv12_border_mode="same",
                conv12_pad="same",

                pool1_pool_size=(2, 2),

                conv2_num_filters=layers_size[1], conv2_filter_size=filter_2, conv2_nonlinearity=activation,
                # conv2_border_mode="same",
                conv2_pad="same",
                conv21_num_filters=layers_size[1], conv21_filter_size=filter_2, conv21_nonlinearity=activation,
                # conv21_border_mode="same",
                conv21_pad="same",
                conv22_num_filters=layers_size[1], conv22_filter_size=filter_2, conv22_nonlinearity=activation,
                # conv22_border_mode="same",
                conv22_pad="same",

                pool2_pool_size=(2, 2),

                conv3_num_filters=layers_size[2], conv3_filter_size=filter_3, conv3_nonlinearity=activation,
                # conv3_border_mode="same",
                conv3_pad="same",
                conv31_num_filters=layers_size[2], conv31_filter_size=filter_3, conv31_nonlinearity=activation,
                # conv31_border_mode="same",
                conv31_pad="same",
                conv32_num_filters=layers_size[2], conv32_filter_size=filter_3, conv32_nonlinearity=activation,
                # conv32_border_mode="same",
                conv32_pad="same",

                pool3_pool_size=(2, 2),

                conv4_num_filters=layers_size[3], conv4_filter_size=filter_4, conv4_nonlinearity=activation,
                # conv4_border_mode="same",
                conv4_pad="same",
                conv41_num_filters=layers_size[3], conv41_filter_size=filter_4, conv41_nonlinearity=activation,
                # conv41_border_mode="same",
                conv41_pad="same",
                conv42_num_filters=1, conv42_filter_size=filter_4, conv42_nonlinearity=activation,
                # conv42_border_mode="same",
                conv42_pad="same",

                unpool1_ds=(2, 2),

                conv5_num_filters=layers_size[4], conv5_filter_size=filter_5, conv5_nonlinearity=activation,
                # conv5_border_mode="same",
                conv5_pad="same",
                conv51_num_filters=layers_size[4], conv51_filter_size=filter_5, conv51_nonlinearity=activation,
                # conv51_border_mode="same",
                conv51_pad="same",
                conv52_num_filters=layers_size[4], conv52_filter_size=filter_5, conv52_nonlinearity=activation,
                # conv52_border_mode="same",
                conv52_pad="same",

                unpool2_ds=(2, 2),

                conv6_num_filters=layers_size[5], conv6_filter_size=filter_6, conv6_nonlinearity=activation,
                # conv6_border_mode="same",
                conv6_pad="same",
                conv61_num_filters=layers_size[5], conv61_filter_size=filter_6, conv61_nonlinearity=activation,
                # conv61_border_mode="same",
                conv61_pad="same",
                conv62_num_filters=layers_size[5], conv62_filter_size=filter_6, conv62_nonlinearity=activation,
                # conv62_border_mode="same",
                conv62_pad="same",

                unpool3_ds=(2, 2),

                conv7_num_filters=layers_size[6], conv7_filter_size=filter_7, conv7_nonlinearity=activation,
                # conv7_border_mode="same",
                conv7_pad="same",
                conv71_num_filters=layers_size[6], conv71_filter_size=filter_7, conv71_nonlinearity=activation,
                # conv71_border_mode="same",
                conv71_pad="same",
                conv72_num_filters=layers_size[6], conv72_filter_size=filter_7, conv72_nonlinearity=activation,
                # conv72_border_mode="same",
                conv72_pad="same",


                conv8_num_filters=1, conv8_filter_size=filter_8, conv8_nonlinearity=last_layer_activation,
                # conv8_border_mode="same",
                conv8_pad="same",

                output_layer_shape=(([0], -1)),

                update_learning_rate=learning_rate,
                update_momentum=update_momentum,
                update=nesterov_momentum,
                train_split=TrainSplit(eval_size=train_valid_split),
                batch_iterator_train=FlipBatchIterator(batch_size=batch_size) if flip_batch else BatchIterator(
                    batch_size=batch_size),
                regression=True,
                max_epochs=epochs,
                verbose=1,
                hiddenLayer_to_output=15)

            return cnn

        def create_cae_2pool_3conv(input_height, input_width):

            cnn = NeuralNet(layers=[
                ('input', layers.InputLayer),
                ('conv1', layers.Conv2DLayer),
                ('conv11', layers.Conv2DLayer),
                ('conv12', layers.Conv2DLayer),

                ('pool1', layers.MaxPool2DLayer),
                ('conv2', layers.Conv2DLayer),
                ('conv21', layers.Conv2DLayer),
                ('conv22', layers.Conv2DLayer),

                ('pool2', layers.MaxPool2DLayer),
                ('conv3', layers.Conv2DLayer),
                ('conv31', layers.Conv2DLayer),
                ('conv32', layers.Conv2DLayer),

                ('unpool1', Unpool2DLayer),
                ('conv4', layers.Conv2DLayer),
                ('conv41', layers.Conv2DLayer),
                ('conv42', layers.Conv2DLayer),

                ('unpool2', Unpool2DLayer),
                ('conv5', layers.Conv2DLayer),
                ('conv51', layers.Conv2DLayer),
                ('conv52', layers.Conv2DLayer),

                ('conv6', layers.Conv2DLayer),
                ('output_layer', ReshapeLayer),
            ],

                input_shape=(None, 1, input_width, input_height),
                # Layer current size - 1x300x140

                conv1_num_filters=layers_size[0], conv1_filter_size=filter_1, conv1_nonlinearity=activation,
                # conv1_border_mode="same",
                conv1_pad="same",
                conv11_num_filters=layers_size[0], conv11_filter_size=filter_1, conv11_nonlinearity=activation,
                # conv11_border_mode="same",
                conv11_pad="same",
                conv12_num_filters=layers_size[0], conv12_filter_size=filter_1, conv12_nonlinearity=activation,
                # conv12_border_mode="same",
                conv12_pad="same",

                pool1_pool_size=(2, 2),

                conv2_num_filters=layers_size[1], conv2_filter_size=filter_2, conv2_nonlinearity=activation,
                # conv2_border_mode="same",
                conv2_pad="same",
                conv21_num_filters=layers_size[1], conv21_filter_size=filter_2, conv21_nonlinearity=activation,
                # conv21_border_mode="same",
                conv21_pad="same",
                conv22_num_filters=layers_size[1], conv22_filter_size=filter_2, conv22_nonlinearity=activation,
                # conv22_border_mode="same",
                conv22_pad="same",

                pool2_pool_size=(2, 2),

                conv3_num_filters=layers_size[2], conv3_filter_size=filter_3, conv3_nonlinearity=activation,
                # conv3_border_mode="same",
                conv3_pad="same",
                conv31_num_filters=layers_size[2], conv31_filter_size=filter_3, conv31_nonlinearity=activation,
                # conv31_border_mode="same",
                conv31_pad="same",
                conv32_num_filters=1, conv32_filter_size=filter_3, conv32_nonlinearity=activation,
                # conv32_border_mode="same",
                conv32_pad="same",

                unpool1_ds=(2, 2),

                conv4_num_filters=layers_size[3], conv4_filter_size=filter_4, conv4_nonlinearity=activation,
                # conv4_border_mode="same",
                conv4_pad="same",
                conv41_num_filters=layers_size[3], conv41_filter_size=filter_4, conv41_nonlinearity=activation,
                # conv41_border_mode="same",
                conv41_pad="same",
                conv42_num_filters=layers_size[3], conv42_filter_size=filter_4, conv42_nonlinearity=activation,
                # conv42_border_mode="same",
                conv42_pad="same",

                unpool2_ds=(2, 2),

                conv5_num_filters=layers_size[4], conv5_filter_size=filter_5, conv5_nonlinearity=activation,
                # conv5_border_mode="same",
                conv5_pad="same",
                conv51_num_filters=layers_size[4], conv51_filter_size=filter_5, conv51_nonlinearity=activation,
                # conv51_border_mode="same",
                conv51_pad="same",
                conv52_num_filters=layers_size[4], conv52_filter_size=filter_5, conv52_nonlinearity=activation,
                # conv52_border_mode="same",
                conv52_pad="same",

                conv6_num_filters=1, conv6_filter_size=filter_6, conv6_nonlinearity=last_layer_activation,
                # conv6_border_mode="same",
                conv6_pad="same",

                output_layer_shape=(([0], -1)),

                update_learning_rate=learning_rate,
                update_momentum=update_momentum,
                update=nesterov_momentum,
                train_split=TrainSplit(eval_size=train_valid_split),
                batch_iterator_train=FlipBatchIterator(batch_size=batch_size) if flip_batch else BatchIterator(
                    batch_size=batch_size),
                regression=True,
                max_epochs=epochs,
                verbose=1,
                hiddenLayer_to_output=11)

            return cnn

        def create_cae_4pool_2conv(input_height, input_width):

            cnn = NeuralNet(layers=[
                ('input', layers.InputLayer),
                ('conv1', layers.Conv2DLayer),
                ('conv11', layers.Conv2DLayer),

                ('pool1', layers.MaxPool2DLayer),

                ('conv2', layers.Conv2DLayer),
                ('conv21', layers.Conv2DLayer),

                ('pool2', layers.MaxPool2DLayer),

                ('conv3', layers.Conv2DLayer),
                ('conv31', layers.Conv2DLayer),

                ('pool3', layers.MaxPool2DLayer),

                ('conv4', layers.Conv2DLayer),
                ('conv41', layers.Conv2DLayer),

                ('pool4', layers.MaxPool2DLayer),

                ('conv5', layers.Conv2DLayer),
                ('conv51', layers.Conv2DLayer),

                ('unpool1', Unpool2DLayer),

                ('conv6', layers.Conv2DLayer),
                ('conv61', layers.Conv2DLayer),

                ('unpool2', Unpool2DLayer),

                ('conv7', layers.Conv2DLayer),
                ('conv71', layers.Conv2DLayer),

                ('unpool3', Unpool2DLayer),

                ('conv8', layers.Conv2DLayer),
                ('conv81', layers.Conv2DLayer),

                ('unpool4', Unpool2DLayer),

                ('conv9', layers.Conv2DLayer),
                ('conv91', layers.Conv2DLayer),

                ('conv10', layers.Conv2DLayer),

                ('output_layer', ReshapeLayer),
            ],

                input_shape=(None, 1, input_width, input_height),
                # Layer current size - 1x320x160

                conv1_num_filters=layers_size[0], conv1_filter_size=filter_1, conv1_nonlinearity=activation,
                # conv1_border_mode="same",
                conv1_pad="same",
                conv11_num_filters=layers_size[0], conv11_filter_size=filter_1, conv11_nonlinearity=activation,
                # conv11_border_mode="same",
                conv11_pad="same",

                pool1_pool_size=(2, 2),

                conv2_num_filters=layers_size[1], conv2_filter_size=filter_2, conv2_nonlinearity=activation,
                # conv2_border_mode="same",
                conv2_pad="same",
                conv21_num_filters=layers_size[1], conv21_filter_size=filter_2, conv21_nonlinearity=activation,
                # conv21_border_mode="same",
                conv21_pad="same",

                pool2_pool_size=(2, 2),

                conv3_num_filters=layers_size[2], conv3_filter_size=filter_3, conv3_nonlinearity=activation,
                # conv3_border_mode="same",
                conv3_pad="same",
                conv31_num_filters=layers_size[2], conv31_filter_size=filter_3, conv31_nonlinearity=activation,
                # conv31_border_mode="same",
                conv31_pad="same",

                pool3_pool_size=(2, 2),

                conv4_num_filters=layers_size[3], conv4_filter_size=filter_4, conv4_nonlinearity=activation,
                # conv4_border_mode="same",
                conv4_pad="same",
                conv41_num_filters=layers_size[3], conv41_filter_size=filter_4, conv41_nonlinearity=activation,
                # conv41_border_mode="same",
                conv41_pad="same",

                pool4_pool_size=(2, 2),

                conv5_num_filters=layers_size[4], conv5_filter_size=filter_5, conv5_nonlinearity=activation,
                # conv5_border_mode="same",
                conv5_pad="same",
                conv51_num_filters=1, conv51_filter_size=filter_5, conv51_nonlinearity=activation,
                # conv51_border_mode="same",
                conv51_pad="same",

                unpool1_ds=(2, 2),

                conv6_num_filters=layers_size[5], conv6_filter_size=filter_6, conv6_nonlinearity=activation,
                # conv6_border_mode="same",
                conv6_pad="same",
                conv61_num_filters=layers_size[5], conv61_filter_size=filter_6, conv61_nonlinearity=activation,
                # conv61_border_mode="same",
                conv61_pad="same",

                unpool2_ds=(2, 2),

                conv7_num_filters=layers_size[6], conv7_filter_size=filter_7, conv7_nonlinearity=activation,
                # conv7_border_mode="same",
                conv7_pad="same",
                conv71_num_filters=layers_size[6], conv71_filter_size=filter_7, conv71_nonlinearity=activation,
                # conv71_border_mode="same",
                conv71_pad="same",

                unpool3_ds=(2, 2),

                conv8_num_filters=layers_size[7], conv8_filter_size=filter_8, conv8_nonlinearity=activation,
                # conv8_border_mode="same",
                conv8_pad="same",
                conv81_num_filters=layers_size[7], conv81_filter_size=filter_8, conv81_nonlinearity=activation,
                # conv81_border_mode="same",
                conv81_pad="same",

                unpool4_ds=(2, 2),

                conv9_num_filters=layers_size[8], conv9_filter_size=filter_9, conv9_nonlinearity=activation,
                # conv9_border_mode="same",
                conv9_pad="same",
                conv91_num_filters=layers_size[8], conv91_filter_size=filter_9, conv91_nonlinearity=activation,
                # conv91_border_mode="same",
                conv91_pad="same",

                conv10_num_filters=1, conv10_filter_size=filter_10, conv10_nonlinearity=last_layer_activation,
                # conv10_border_mode="same",
                conv10_pad="same",

                output_layer_shape=(([0], -1)),

                update_learning_rate=learning_rate,
                update_momentum=update_momentum,
                update=nesterov_momentum,
                train_split=TrainSplit(eval_size=train_valid_split),
                batch_iterator_train=FlipBatchIterator(batch_size=batch_size) if flip_batch else BatchIterator(
                    batch_size=batch_size),
                regression=True,
                max_epochs=epochs,
                verbose=1,
                hiddenLayer_to_output=14)

            return cnn

        def create_cae_3pool_2conv(input_height, input_width):

            cnn = NeuralNet(layers=[
                ('input', layers.InputLayer),
                ('conv1', layers.Conv2DLayer),
                ('conv11', layers.Conv2DLayer),

                ('pool1', layers.MaxPool2DLayer),

                ('conv2', layers.Conv2DLayer),
                ('conv21', layers.Conv2DLayer),

                ('pool2', layers.MaxPool2DLayer),

                ('conv3', layers.Conv2DLayer),
                ('conv31', layers.Conv2DLayer),

                ('pool3', layers.MaxPool2DLayer),

                ('conv4', layers.Conv2DLayer),
                ('conv41', layers.Conv2DLayer),

                ('unpool1', Unpool2DLayer),

                ('conv5', layers.Conv2DLayer),
                ('conv51', layers.Conv2DLayer),

                ('unpool2', Unpool2DLayer),

                ('conv6', layers.Conv2DLayer),
                ('conv61', layers.Conv2DLayer),

                ('unpool3', Unpool2DLayer),

                ('conv7', layers.Conv2DLayer),
                ('conv71', layers.Conv2DLayer),

                ('conv8', layers.Conv2DLayer),

                ('output_layer', ReshapeLayer),
            ],

                input_shape=(None, 1, input_width, input_height),
                # Layer current size - 1x320x160

                conv1_num_filters=layers_size[0], conv1_filter_size=filter_1, conv1_nonlinearity=activation,
                # conv1_border_mode="same",
                conv1_pad="same",
                conv11_num_filters=layers_size[0], conv11_filter_size=filter_1, conv11_nonlinearity=activation,
                # conv11_border_mode="same",
                conv11_pad="same",

                pool1_pool_size=(2, 2),

                conv2_num_filters=layers_size[1], conv2_filter_size=filter_2, conv2_nonlinearity=activation,
                # conv2_border_mode="same",
                conv2_pad="same",
                conv21_num_filters=layers_size[1], conv21_filter_size=filter_2, conv21_nonlinearity=activation,
                # conv21_border_mode="same",
                conv21_pad="same",

                pool2_pool_size=(2, 2),

                conv3_num_filters=layers_size[2], conv3_filter_size=filter_3, conv3_nonlinearity=activation,
                # conv3_border_mode="same",
                conv3_pad="same",
                conv31_num_filters=layers_size[2], conv31_filter_size=filter_3, conv31_nonlinearity=activation,
                # conv31_border_mode="same",
                conv31_pad="same",

                pool3_pool_size=(2, 2),

                conv4_num_filters=layers_size[3], conv4_filter_size=filter_4, conv4_nonlinearity=activation,
                # conv4_border_mode="same",
                conv4_pad="same",
                conv41_num_filters=1, conv41_filter_size=filter_4, conv41_nonlinearity=activation,
                # conv41_border_mode="same",
                conv41_pad="same",

                unpool1_ds=(2, 2),

                conv5_num_filters=layers_size[4], conv5_filter_size=filter_5, conv5_nonlinearity=activation,
                # conv5_border_mode="same",
                conv5_pad="same",
                conv51_num_filters=layers_size[4], conv51_filter_size=filter_5, conv51_nonlinearity=activation,
                # conv51_border_mode="same",
                conv51_pad="same",

                unpool2_ds=(2, 2),

                conv6_num_filters=layers_size[5], conv6_filter_size=filter_6, conv6_nonlinearity=activation,
                # conv6_border_mode="same",
                conv6_pad="same",
                conv61_num_filters=layers_size[5], conv61_filter_size=filter_6, conv61_nonlinearity=activation,
                # conv61_border_mode="same",
                conv61_pad="same",

                unpool3_ds=(2, 2),

                conv7_num_filters=layers_size[6], conv7_filter_size=filter_7, conv7_nonlinearity=activation,
                # conv7_border_mode="same",
                conv7_pad="same",
                conv71_num_filters=layers_size[6], conv71_filter_size=filter_7, conv71_nonlinearity=activation,
                # conv71_border_mode="same",
                conv71_pad="same",

                conv8_num_filters=1, conv8_filter_size=filter_8, conv8_nonlinearity=last_layer_activation,
                # conv8_border_mode="same",
                conv8_pad="same",

                output_layer_shape=(([0], -1)),

                update_learning_rate=learning_rate,
                update_momentum=update_momentum,
                update=nesterov_momentum,
                train_split=TrainSplit(eval_size=train_valid_split),
                batch_iterator_train=FlipBatchIterator(batch_size=batch_size) if flip_batch else BatchIterator(
                    batch_size=batch_size),
                regression=True,
                max_epochs=epochs,
                verbose=1,
                hiddenLayer_to_output=11)

            return cnn

        def create_cae_2pool_2conv(input_height, input_width):

            cnn = NeuralNet(layers=[
                ('input', layers.InputLayer),
                ('conv1', layers.Conv2DLayer),
                ('conv11', layers.Conv2DLayer),

                ('pool1', layers.MaxPool2DLayer),
                ('conv2', layers.Conv2DLayer),
                ('conv21', layers.Conv2DLayer),

                ('pool2', layers.MaxPool2DLayer),
                ('conv3', layers.Conv2DLayer),
                ('conv31', layers.Conv2DLayer),

                ('unpool1', Unpool2DLayer),
                ('conv4', layers.Conv2DLayer),
                ('conv41', layers.Conv2DLayer),

                ('unpool2', Unpool2DLayer),
                ('conv5', layers.Conv2DLayer),
                ('conv51', layers.Conv2DLayer),

                ('conv6', layers.Conv2DLayer),
                ('output_layer', ReshapeLayer),
            ],

                input_shape=(None, 1, input_width, input_height),
                # Layer current size - 1x300x140

                conv1_num_filters=layers_size[0], conv1_filter_size=filter_1, conv1_nonlinearity=activation,
                # conv1_border_mode="same",
                conv1_pad="same",
                conv11_num_filters=layers_size[0], conv11_filter_size=filter_1, conv11_nonlinearity=activation,
                # conv11_border_mode="same",
                conv11_pad="same",

                pool1_pool_size=(2, 2),

                conv2_num_filters=layers_size[1], conv2_filter_size=filter_2, conv2_nonlinearity=activation,
                # conv2_border_mode="same",
                conv2_pad="same",
                conv21_num_filters=layers_size[1], conv21_filter_size=filter_2, conv21_nonlinearity=activation,
                # conv21_border_mode="same",
                conv21_pad="same",

                pool2_pool_size=(2, 2),

                conv3_num_filters=layers_size[2], conv3_filter_size=filter_3, conv3_nonlinearity=activation,
                # conv3_border_mode="same",
                conv3_pad="same",
                conv31_num_filters=1, conv31_filter_size=filter_3, conv31_nonlinearity=activation,
                # conv31_border_mode="same",
                conv31_pad="same",

                unpool1_ds=(2, 2),

                conv4_num_filters=layers_size[3], conv4_filter_size=filter_4, conv4_nonlinearity=activation,
                # conv4_border_mode="same",
                conv4_pad="same",
                conv41_num_filters=layers_size[3], conv41_filter_size=filter_4, conv41_nonlinearity=activation,
                # conv41_border_mode="same",
                conv41_pad="same",

                unpool2_ds=(2, 2),

                conv5_num_filters=layers_size[4], conv5_filter_size=filter_5, conv5_nonlinearity=activation,
                # conv5_border_mode="same",
                conv5_pad="same",
                conv51_num_filters=layers_size[4], conv51_filter_size=filter_5, conv51_nonlinearity=activation,
                # conv51_border_mode="same",
                conv51_pad="same",

                conv6_num_filters=1, conv6_filter_size=filter_6, conv6_nonlinearity=last_layer_activation,
                # conv6_border_mode="same",
                conv6_pad="same",

                output_layer_shape=(([0], -1)),

                update_learning_rate=learning_rate,
                update_momentum=update_momentum,
                update=nesterov_momentum,
                train_split=TrainSplit(eval_size=train_valid_split),
                batch_iterator_train=FlipBatchIterator(batch_size=batch_size) if flip_batch else BatchIterator(
                    batch_size=batch_size),
                regression=True,
                max_epochs=epochs,
                verbose=1,
                hiddenLayer_to_output=8)

            return cnn

        if number_pooling_layers == 2:
            if filters_type == 3:
                filter_1 = (3, 3)
                filter_2 = (3, 3)
                filter_3 = (3, 3)
                filter_4 = (3, 3)
                filter_5 = (3, 3)
                filter_6 = (3, 3)
                filter_7 = 0
                filter_8 = 0
                filter_9 = 0
                filter_10 = 0
            elif filters_type == 5:
                filter_1 = (5, 5)
                filter_2 = (5, 5)
                filter_3 = (5, 5)
                filter_4 = (5, 5)
                filter_5 = (5, 5)
                filter_6 = (5, 5)
                filter_7 = 0
                filter_8 = 0
                filter_9 = 0
                filter_10 = 0
            elif filters_type == 7:
                filter_1 = (7, 7)
                filter_2 = (5, 5)
                filter_3 = (3, 3)
                filter_4 = (5, 5)
                filter_5 = (7, 7)
                filter_6 = (5, 5)
                filter_7 = 0
                filter_8 = 0
                filter_9 = 0
                filter_10 = 0
            elif filters_type == 9:
                filter_1 = (9, 9)
                filter_2 = (7, 7)
                filter_3 = (5, 5)
                filter_4 = (7, 7)
                filter_5 = (9, 9)
                filter_6 = (5, 5)
                filter_7 = 0
                filter_8 = 0
                filter_9 = 0
                filter_10 = 0
            elif filters_type == 11:
                filter_1 = (11, 11)
                filter_2 = (9, 9)
                filter_3 = (7, 7)
                filter_4 = (9, 9)
                filter_5 = (11, 11)
                filter_6 = (5, 5)
                filter_7 = 0
                filter_8 = 0
                filter_9 = 0
                filter_10 = 0

            if number_conv_layers == 4:
                cnn = create_cae_2pool_4conv(input_height, input_width)
            elif number_conv_layers == 3:
                cnn = create_cae_2pool_3conv(input_height, input_width)
            elif number_conv_layers == 2:
                cnn = create_cae_2pool_2conv(input_height, input_width)

        elif number_pooling_layers == 3:
            if filters_type == 3:
                filter_1 = (3, 3)
                filter_2 = (3, 3)
                filter_3 = (3, 3)
                filter_4 = (3, 3)
                filter_5 = (3, 3)
                filter_6 = (3, 3)
                filter_7 = (3, 3)
                filter_8 = (3, 3)
                filter_9 = 0
                filter_10 = 0
            elif filters_type == 5:
                filter_1 = (5, 5)
                filter_2 = (5, 5)
                filter_3 = (5, 5)
                filter_4 = (5, 5)
                filter_5 = (5, 5)
                filter_6 = (5, 5)
                filter_7 = (5, 5)
                filter_8 = (5, 5)
                filter_9 = 0
                filter_10 = 0
            elif filters_type == 7:
                filter_1 = (7, 7)
                filter_2 = (5, 5)
                filter_3 = (5, 5)
                filter_4 = (3, 3)
                filter_5 = (5, 5)
                filter_6 = (5, 5)
                filter_7 = (7, 7)
                filter_8 = (5, 5)
                filter_9 = 0
                filter_10 = 0
            elif filters_type == 9:
                filter_1 = (9, 9)
                filter_2 = (7, 7)
                filter_3 = (7, 7)
                filter_4 = (5, 5)
                filter_5 = (3, 3)
                filter_6 = (5, 5)
                filter_7 = (7, 7)
                filter_8 = (7, 7)
                filter_9 = 0
                filter_10 = 0
            elif filters_type == 11:
                filter_1 = (11, 11)
                filter_2 = (9, 9)
                filter_3 = (7, 7)
                filter_4 = (5, 5)
                filter_5 = (5, 5)
                filter_6 = (7, 7)
                filter_7 = (11, 11)
                filter_8 = (5, 5)
                filter_9 = 0
                filter_10 = 0

            if number_conv_layers == 4:
                cnn = create_cae_3pool_4conv(input_height, input_width)
            elif number_conv_layers == 3:
                cnn = create_cae_3pool_3conv(input_height, input_width)
            elif number_conv_layers == 2:
                cnn = create_cae_3pool_2conv(input_height, input_width)

        elif number_pooling_layers == 4:
            if filters_type == 3:
                filter_1 = (3, 3)
                filter_2 = (3, 3)
                filter_3 = (3, 3)
                filter_4 = (3, 3)
                filter_5 = (3, 3)
                filter_6 = (3, 3)
                filter_7 = (3, 3)
                filter_8 = (3, 3)
                filter_9 = (3, 3)
                filter_10 = (3, 3)
            elif filters_type == 5:
                filter_1 = (5, 5)
                filter_2 = (5, 5)
                filter_3 = (5, 5)
                filter_4 = (5, 5)
                filter_5 = (5, 5)
                filter_6 = (5, 5)
                filter_7 = (5, 5)
                filter_8 = (5, 5)
                filter_9 = (5, 5)
                filter_10 = (5, 5)
            elif filters_type == 7:
                filter_1 = (7, 7)
                filter_2 = (5, 5)
                filter_3 = (5, 5)
                filter_4 = (5, 5)
                filter_5 = (3, 3)
                filter_6 = (3, 3)
                filter_7 = (5, 5)
                filter_8 = (5, 5)
                filter_9 = (7, 7)
                filter_10 = (5, 5)
            elif filters_type == 9:
                filter_1 = (9, 9)
                filter_2 = (7, 7)
                filter_3 = (7, 7)
                filter_4 = (5, 5)
                filter_5 = (3, 3)
                filter_6 = (5, 5)
                filter_7 = (7, 7)
                filter_8 = (7, 7)
                filter_9 = (9, 9)
                filter_10 = (5, 5)
            elif filters_type == 11:
                filter_1 = (11, 11)
                filter_2 = (9, 9)
                filter_3 = (7, 7)
                filter_4 = (5, 5)
                filter_5 = (3, 3)
                filter_6 = (5, 5)
                filter_7 = (7, 7)
                filter_8 = (9, 9)
                filter_9 = (11, 11)
                filter_10 = (5, 5)

            if number_conv_layers == 4:
                cnn = create_cae_4pool_4conv(input_height, input_width)
            elif number_conv_layers == 3:
                cnn = create_cae_4pool_3conv(input_height, input_width)
            elif number_conv_layers == 2:
                cnn = create_cae_4pool_2conv(input_height, input_width)

        write_filters_to_files(filter_1, filter_10, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7,
                               filter_8, filter_9)
        return cnn

    def write_filters_to_files(filter_1, filter_10, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7,
                               filter_8, filter_9):
        output_file.write("filters_info:\n" + str(filter_1) + "\n")
        output_file.write(str(filter_2) + "\n")
        output_file.write(str(filter_3) + "\n")
        output_file.write(str(filter_4) + "\n")
        output_file.write(str(filter_5) + "\n")
        output_file.write(str(filter_6) + "\n")
        if filter_7 is not 0:
            output_file.write(str(filter_7) + "\n")
            output_file.write(str(filter_8) + "\n")
            if filter_9 is not 0:
                output_file.write(str(filter_9) + "\n")
                output_file.write(str(filter_10) + "\n\n")

        results_file.write(str(filter_1) + "\t" + str((filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7, filter_8,
                                      filter_9, filter_10)) + "\t")

    def train_cae(cnn, x_train, x_out):

        x_train *= np.random.binomial(1, 1 - dropout_percent, size=x_train.shape)
        print('Training CAE with ', x_train.shape[0], ' samples')
        cnn.fit(x_train, x_out)

        try:
            save_example_images(x_out, cnn)
        except Exception as e:
            print (e)
            print (e.message)

        return cnn

    def save_example_images(x_out, cnn, x_train=None):
        print("Saving some images....")
        for i in range(10):
            index = np.random.randint(x_out.shape[0])
            print(index)
            image_sample = x_out[index]
            image_sample = image_sample.astype(np.float32).reshape((-1, 1, input_width, input_height))

            print('Predicting ', index, ' sample through CAE')
            X_pred = cnn.predict(image_sample).reshape(-1, input_height, input_width)  # * sigma + mu

            def get_picture_array(X):
                array = np.rint(X * 256).astype(np.int).reshape(input_height, input_width)
                array = np.clip(array, a_min=0, a_max=255)
                return array.repeat(4, axis=0).repeat(4, axis=1).astype(np.uint8())

            original_image = Image.fromarray(get_picture_array(x_out[index]))
            # original_image.save(folder_path + 'original' + str(index) + '.png', format="PNG")
            #
            # array = np.rint(trian_last_hiddenLayer[index] * 256).astype(np.int).reshape(input_height/2, input_width/2)
            # array = np.clip(array, a_min=0, a_max=255)
            # encode_image = Image.fromarray(array.repeat(4, axis=0).repeat(4, axis=1).astype(np.uint8()))
            # encode_image.save(folder_path + 'encode' + str(index) + '.png', format="PNG")

            new_size = (original_image.size[0] * 2, original_image.size[1])
            new_im = Image.new('L', new_size)
            new_im.paste(original_image, (0, 0))
            pred_image = Image.fromarray(get_picture_array(X_pred))
            pred_image.save(folder_path + 'pred' + str(index) + '.png', format="PNG")
            new_im.paste(pred_image, (original_image.size[0], 0))
            new_im.save(folder_path + 'origin_prediction-' + str(index) + '.png', format="PNG")


            # noise_image = Image.fromarray(get_picture_array(x_train[index]))
            # new_im.paste(noise_image, (original_image.size[0] * 2, 0))
            # new_im.save(folder_path + 'origin_prediction_noise-' + str(index) + '.png', format="PNG")

            # diff = ImageChops.difference(original_image, pred_image)
            # diff = diff.convert('L')
            # diff.save(folder_path + 'diff' + str(index) + '.png', format="PNG")

            # # plt.imshow(new_im)
            # new_size = (original_image.size[0] * 2, original_image.size[1])
            # new_im = Image.new('L', new_size)
            # new_im.paste(original_image, (0, 0))
            # pred_image = Image.fromarray(get_picture_array(X_train, index))
            # # pred_image.save(folder_path + 'noisyInput' + str(index) + '.png', format="PNG")
            # new_im.paste(pred_image, (original_image.size[0], 0))
            # new_im.save(folder_path+'origin_VS_noise-'+str(index)+'.png', format="PNG")
            # # plt.imshow(new_im)

    def write_output_file(train_history, layer_info):
        # save the network's parameters
        output_file.write("Validation set error: " + str(train_history[-1]['valid_accuracy']) + "\n\n")
        results_file.write(str(train_history[-1]['valid_accuracy']) + "\t")

        output_file.write("Training NN on: " + ("20 Top Categories\n" if 20 == categories else "Article Categories\n"))
        output_file.write("Learning rate: " + str(learning_rate) + "\n")
        results_file.write(str(learning_rate) + "\t")
        output_file.write(("Momentum: " + str(update_momentum) + "\n") if update_rho is None else (
            "Decay Factor: " + str(update_rho) + "\n"))
        results_file.write(str(update_momentum) + "\t")
        output_file.write(("FlipBatcherIterater" if flip_batch else "BatchIterator") + " with batch: " + str(batch_size) + "\n")
        results_file.write("FlipBatcherIterater\t" + str(batch_size) + "\t")
        output_file.write("Num epochs: " + str(epochs) + "\n")
        results_file.write(str(epochs) + "\t")
        output_file.write("Layers size: " + str(layers_size) + "\n\n")
        results_file.write(str(layers_size) + "\t")
        output_file.write("Activation func: " + ("Rectify" if activation is None else str(activation)) + "\n")
        results_file.write(("Rectify" if activation is None else str(activation)) + "\t")
        output_file.write(
            "Last layer activation func: " + ("Rectify" if last_layer_activation is None else str(last_layer_activation)) + "\n")
        results_file.write(("Rectify" if last_layer_activation is None else str(last_layer_activation)) + "\t")
        #         output_file.write("Multiple Positives by: " + str(multiple_positives) + "\n")
        output_file.write("Number of images for training: " + str(amount_train) + "\n")
        results_file.write(str(amount_train) + "\t")
        output_file.write("Number of negative images in svm: " + str(svm_negative_amount) + "\n")
        output_file.write("Dropout noise precent: " + str(dropout_percent * 100) + "%\n")
        results_file.write(str(dropout_percent * 100) + "%\t")
        output_file.write("Train/validation split: " + str(train_valid_split) + "\n")
        results_file.write(str(train_valid_split) + "\t")
        output_file.write("shuffle_input: " + str(shuffle_input) + "\n")
        results_file.write(str(shuffle_input) + "\t")
        output_file.write("zero_meaning: " + str(zero_meaning) + "\n\n")
        results_file.write(str(zero_meaning) + "\t")

        output_file.write("history: " + str(train_history) + "\n\n")
        results_file.write(str(train_history) + "\t")
        output_file.write("layer_info:\n" + str(layer_info) + "\n")
        results_file.write("[" + str(layer_info).replace("\n", ",") + "]\t")
        output_file.write("Run time[minutes] is: " + str(run_time) + "\n")

        output_file.flush()
        results_file.write(str(time.ctime()) + "\t")
        results_file.write(folder_path + "\n")
        results_file.flush()

    if loadedData is None:
        train_x, train_y, test_x, test_y = load2d(categories, output_file, input_width, input_height, amount_train,
                                                  multiple_positives, dropout_percent, end_index=amount_train)
    else:
        data = loadedData
        train_x, train_y, test_x, test_y = data

    if zero_meaning:
        # train_x = train_x.astype(np.float64)
        mu, sigma = np.mean(train_x.flatten()), np.std(train_x.flatten())
        print("Mean- ", mu)
        print("Std- ", sigma)
        train_x = (train_x - mu) / sigma

    x_train = train_x.astype(np.float32).reshape((-1, 1, input_width, input_height))
    x_flat = x_train.reshape((x_train.shape[0], -1))

    if LOAD_CAE_PATH is None:
        start_time = time.clock()
        print ("Start time: ", time.ctime())
        cae = create_cae()
        cae = train_cae(cae, x_train[:amount_train], x_flat[:amount_train])
        run_time = (time.clock() - start_time) / 60.
        write_output_file(cae.train_history_, PrintLayerInfo._get_layer_info_plain(cae))
        print ("Learning took (min)- ", run_time)
        valid_accuracy = cae.train_history_[-1]['valid_accuracy']
        if valid_accuracy > 0.05:
            return valid_accuracy

        save_cnn(cae, folder_path)
    else:
        cae = load_network(LOAD_CAE_PATH)
        valid_accuracy = cae.train_history_[-1]['valid_accuracy']

    get_auc_score(cae, output_file, results_file, svm_negative_amount, train_y, x_train, folder_path)

    return valid_accuracy


def get_auc_score(cnn, output_file, results_file, svm_negative_amount, train_y, x_train, folder_path):
    try:
        print("Running SVM")
        print("     Start time: ", time.ctime())
        errors, aucs = run_svm(cnn, X_train=x_train, labels=train_y, svm_negative_amount=svm_negative_amount,
                               folder_path=folder_path)
        print("NN AUC", errors)
        print("SVM AUC", aucs)
        output_file.write("NN auc: " + str(errors) + "\n")
        output_file.write("SVM auc: " + str(aucs) + "\n")
        results_file.write(str(np.average(aucs)) + "\t" + str(aucs) + "\n")

        output_file.flush()
        results_file.flush()
    except Exception as e:
        print(e)
        print(e.message)


def save_cnn(cnn, folder_path):
    try:
        sys.setrecursionlimit(10000)
        print("Trying to pickle network... ")
        pickle.dump(cnn, open(folder_path + "nn.pkl", 'w'))
        pickle.dump(cnn, open(folder_path + "nn_b.pkl", 'wb'))
        print("    Done pickle network... ")
    except Exception as e:
        print(" Could not pickle network... ")
        print(e)
        print(e.message)


def load_network(folder_path):
    try:
        sys.setrecursionlimit(10000)
        print("Trying to load network... ")
        return pickle.load(open(folder_path + "nn.pkl", 'r'))
    except Exception as e:
        print(" Could not load network... ")
        print(e)
        print(e.message)


def run_all():
    if platform.dist()[0]:
        print ("Running in Ubuntu")
    else:
        print ("Running in Windows")

    print(theano.sandbox.cuda.dnn_available())

    num_labels = 15 #164 #TEST: change 15
    amount_train = 16351
    svm_negative_amount = 200
    input_noise_rate = 0.2
    zero_meaning = False
    epochs = 20
    folder_name = "CAE_" + str(amount_train) + "_test_nn-" + str(time.time())

    steps = [
        [5000, 10000, 16352],
        [5000, 10000, 16352],
        [5000, 10000, 16352],
        [5000, 10000, 16352],
        [5000, 10000, 16352],
        [5000, 11000, 16352],
        [5000, 10000, 15000, 16352],
        [4000, 8000, 12000, 16000, 16352]
    ]
    image_width = [160, 160, 160, 200, 240, 300, 320, 320]
    image_height = [80, 80, 100, 120, 120, 140, 160, 200]
    number_pooling_layers = [3, 2, 2, 2, 2, 2, 3, 3]
    layers_size = [
        [16, 16, 16, 16, 16, 16, 16, 16, 16],
        [8, 8, 8, 8, 8, 8, 8, 8, 8],
        [32, 64, 128, 64, 32],
        [16, 32, 32, 64, 32, 32, 16]
        ]

    for num_filters_index in range(0, 1, 1):
        try:
            for lr in range(1, 4, 1):
                try:
                    for f in range(2, 0, -1): # for f in range(0, 3, 1):
                        try:
                            for number_conv_layers in range(4, 1, -2): #2
                                try:
                                    for input_size_index in range(4, 7, 1):  # 5
                                        try:
                                            for num_images in range(0, 9, 1):  # 5
                                                input_size_index = 5 #Test: change
                                                # num_filters_index = 0 #Test: change
                                                data = load2d(batch_index=1, num_labels=num_labels, TRAIN_PRECENT=1,
                                                              steps=steps[input_size_index],
                                                              image_width=image_width[input_size_index],
                                                              image_height=image_height[input_size_index])
                                                learning_rate = 0.04 + 0.005 * lr
                                                filter_type_index = 11 - 4 * f
                                                print("run number conv layers- ", number_conv_layers)
                                                print("run Filter type #", filter_type_index)
                                                print("run Filter number index #", num_filters_index)
                                                print("run Learning rate- ", learning_rate)
                                                try:
                                                    run(layers_size=layers_size[num_filters_index], epochs=epochs,
                                                        learning_rate=learning_rate,
                                                        update_momentum=0.9,
                                                        number_pooling_layers=number_pooling_layers[input_size_index],
                                                        dropout_percent=input_noise_rate, loadedData=data,
                                                        folder_name=folder_name,
                                                        amount_train=amount_train - num_images*2000,
                                                        number_conv_layers=number_conv_layers,
                                                        zero_meaning=zero_meaning, activation=None,
                                                        last_layer_activation=tanh,
                                                        filters_type=filter_type_index,
                                                        train_valid_split=0.001 + 0.002*num_images,
                                                        input_width=image_width[input_size_index],
                                                        input_height=image_height[input_size_index],
                                                        svm_negative_amount=svm_negative_amount, batch_size=32)
                                                    return
                                                except Exception as e:
                                                    print("failed Filter type #", filter_type_index)
                                                    print("failed number conv layers- ", number_conv_layers)
                                                    print("failed Filter number index #", num_filters_index)
                                                    print("failed Learning rate- ", learning_rate)
                                                    print(e)
                                                    print(e.message)
                                        except Exception as e:
                                            print(e)
                                            print(e.message)
                                except Exception as e:
                                    print(e)
                                    print(e.message)

                        except Exception as e:
                            print(e)
                            print(e.message)
                except Exception as e:
                        print(e)
                        print(e.message)
        except Exception as e:
            print(e)
            print(e.message)


if __name__ == "__main__":
    # os.environ["DISPLAY"] = ":99"
    #Test: change
    LOAD_CAE_PATH = "C:\devl\work\ISH_Lasagne\src\DeepLearning\results_dae\CAE_16351_different_sizes-1489653160.29\run_4\\"
    LOAD_CAE_PATH = LOAD_CAE_PATH.replace("\r", "\\r")

    run_all()
