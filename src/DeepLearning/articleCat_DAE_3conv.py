from __future__ import print_function
import os
import time
import sys

from lasagne.nonlinearities import tanh, LeakyRectify
from lasagne import layers
from lasagne.objectives import aggregate, squared_error
from lasagne.updates import nesterov_momentum
from lasagne.updates import rmsprop
from  matplotlib import pyplot
import theano
from autoencoder import DenoisingAutoencoder

# from theano.sandbox.neighbours import neibs2images
# from lasagne.nonlinearities import tanh
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import precision_score
# import urllib
from IPython.display import Image as IPImage
from PIL import Image, ImageChops
import matplotlib.pyplot as plt
from featursForSvm import run_svm

# from nolearn.lasagne import Unpool2DLayer


import cPickle as pickle
import gzip, cPickle
import platform

from logistic_sgd import load_data
from nolearn.lasagne import BatchIterator
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import PrintLayerInfo
from nolearn.lasagne import TrainSplit
import numpy as np
from pickleImages import runPickleImages
from runMySvm import runSvm
from nnClassifier import runNNclassifier
from writeDataForSVM import writeDataToFile
from runCrossValidationSvm import runCrossSvm, runAllLabels
# from predictFromCNN import runPredecitNN

from rl_dae.SDA_layers import StackedDA 
from runLibSvm import runLibSvm
### this is really dumb, current nolearn doesnt play well with lasagne,
### so had to manually copy the file I wanted to this folder
from shape import ReshapeLayer
from theano.tensor.shared_randomstreams import RandomStreams

FILE_SEPARATOR = "/"
counter = 0
isUbuntu = False

def load2d(num_labels, batch_index=1, outputFile=None, input_width=300, input_height=140, end_index=16351, MULTI_POSITIVES=20,
           dropout_percent=0.1, data_set='ISH.pkl.gz', toShuffleInput = False, withZeroMeaning = False, TRAIN_PRECENT=0.8):
    print ('loading data...')

    data_sets, svm_data, svm_label = load_data(data_set, batch_index=batch_index, withSVM=400, toShuffleInput=toShuffleInput,
                                               withZeroMeaning=withZeroMeaning, end_index=end_index,
                                               MULTI_POSITIVES=MULTI_POSITIVES, dropout_percent=dropout_percent,
                                               labelset=num_labels, TRAIN_DATA_PRECENT=TRAIN_PRECENT)

    train_set_x, train_set_y = data_sets[0]
#     valid_set_x, valid_set_y = data_sets[1]
    test_set_x, test_set_y = data_sets[2]
    
#     train_set_x = train_set_x.reshape(-1, 1, input_width, input_height)
# #         valid_set_x = valid_set_x.reshape(-1, 1, input_width, input_height)
#     test_set_x = test_set_x.reshape(-1, 1, input_width, input_height)

    print(train_set_x.shape[0], 'train samples')
    # if outputFile is not None:
    #     outputFile.write("Number of training examples: "+str(train_set_x.shape[0]) + "\n\n")
    return (train_set_x, train_set_y, test_set_x, test_set_y), svm_data, svm_label

# <codecell>

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

# <codecell>

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

def run(loadedData=None, learning_rate=0.04, update_momentum=0.9, update_rho=None, epochs=15,
        input_width=300, input_height=140, train_valid_split=0.2, multiple_positives=20, flip_batch=True,
        dropout_percent=0.1, end_index=16351, activation=None, last_layer_activation=None, batch_size=32,
        layers_size=[5, 10, 20, 40], shuffle_input=False, zero_meaning=False, filters_type=3,
        input_noise_rate=0.3, pre_train_epochs=1, softmax_train_epochs=2, fine_tune_epochs=2,
        categories=15, folder_name="default", dataset='withOutDataSet'):

    global counter
    folder_path = "results_dae"+FILE_SEPARATOR + folder_name + FILE_SEPARATOR + "run_" + str(counter) + FILE_SEPARATOR
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    PARAMS_FILE_NAME = folder_path + "parameters.txt"
    HIDDEN_LAYER_OUTPUT_FILE_NAME = folder_path + "hiddenLayerOutput.pkl.gz"
    FIG_FILE_NAME = folder_path + "fig"
    PICKLES_NET_FILE_NAME = folder_path + "picklesNN.pkl.gz"
    SVM_FILE_NAME = folder_path + "svmData.txt"
    LOG_FILE_NAME = folder_path + "message.log"

    All_Results_FIle = "results_dae"+FILE_SEPARATOR + "all_results.txt"


    #     old_stdout = sys.stdout
    #     print "less",LOG_FILE_NAME
    log_file = False  #open(LOG_FILE_NAME, "w")
    #     sys.stdout = log_file

    counter += 1

    output_file = open(PARAMS_FILE_NAME, "w")
    results_file = open(All_Results_FIle, "a")

    if filters_type == 3:
        filter_1 = (3, 3)
        filter_2 = (3, 3)
        filter_3 = (3, 3)
        filter_4 = (3, 3)
        filter_5 = (3, 3)
        filter_6 = (3, 3)
    elif filters_type == 5:
        filter_1 = (5, 5)
        filter_2 = (5, 5)
        filter_3 = (5, 5)
        filter_4 = (5, 5)
        filter_5 = (5, 5)
        filter_6 = (5, 5)
    elif filters_type == 7:
        filter_1 = (7, 7)
        filter_2 = (7, 7)
        filter_3 = (5, 5)
        filter_4 = (7, 7)
        filter_5 = (7, 7)
        filter_6 = (5, 5)
    elif filters_type == 9:
        filter_1 = (9, 9)
        filter_2 = (7, 7)
        filter_3 = (5, 5)
        filter_4 = (7, 7)
        filter_5 = (9, 9)
        filter_6 = (5, 5)

    elif filters_type == 11:
        filter_1 = (11, 11)
        filter_2 = (9, 9)
        filter_3 = (7, 7)
        filter_4 = (9, 9)
        filter_5 = (11, 11)
        filter_6 = (7, 7)

    def createCSAE(input_height, input_width, X_train, X_out):

        X_train *= np.random.binomial(1, 1-dropout_percent, size=X_train.shape)

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
            hiddenLayer_to_output=-13)

        cnn.fit(X_train, X_out)

        try:
            pickle.dump(cnn, open(folder_path + 'conv_ae.pkl', 'w'))
            # cnn = pickle.load(open(folder_path + 'conv_ae.pkl','r'))
            cnn.save_weights_to(folder_path + 'conv_ae.np')
        except:
            print ("Could not pickle cnn")

        X_pred = cnn.predict(X_train).reshape(-1, input_height, input_width)  # * sigma + mu
        # # X_pred = np.rint(X_pred).astype(int)
        # # X_pred = np.clip(X_pred, a_min=0, a_max=255)
        # # X_pred = X_pred.astype('uint8')
        #
        # try:
        #     trian_last_hiddenLayer = cnn.output_hiddenLayer(X_train)
        #     # test_last_hiddenLayer = cnn.output_hiddenLayer(test_x)
        #     pickle.dump(trian_last_hiddenLayer, open(folder_path + 'encode.pkl', 'w'))
        # except:
        #     print "Could not save encoded images"

        print ("Saving some images....")
        for i in range(10):
            index = np.random.randint(X_train.shape[0])
            print (index)

            def get_picture_array(X, index):
                array = np.rint(X[index] * 256).astype(np.int).reshape(input_height, input_width)
                array = np.clip(array, a_min=0, a_max=255)
                return array.repeat(4, axis=0).repeat(4, axis=1).astype(np.uint8())

            original_image = Image.fromarray(get_picture_array(X_out, index))
            # original_image.save(folder_path + 'original' + str(index) + '.png', format="PNG")
            #
            # array = np.rint(trian_last_hiddenLayer[index] * 256).astype(np.int).reshape(input_height/2, input_width/2)
            # array = np.clip(array, a_min=0, a_max=255)
            # encode_image = Image.fromarray(array.repeat(4, axis=0).repeat(4, axis=1).astype(np.uint8()))
            # encode_image.save(folder_path + 'encode' + str(index) + '.png', format="PNG")

            new_size = (original_image.size[0] * 3, original_image.size[1])
            new_im = Image.new('L', new_size)
            new_im.paste(original_image, (0, 0))
            pred_image = Image.fromarray(get_picture_array(X_pred, index))
            # pred_image.save(folder_path + 'pred' + str(index) + '.png', format="PNG")
            new_im.paste(pred_image, (original_image.size[0], 0))

            noise_image = Image.fromarray(get_picture_array(X_train, index))
            new_im.paste(noise_image, (original_image.size[0]*2, 0))
            new_im.save(folder_path+'origin_prediction_noise-'+str(index)+'.png', format="PNG")

            # diff = ImageChops.difference(original_image, pred_image)
            # diff = diff.convert('L')
            # diff.save(folder_path + 'diff' + str(index) + '.png', format="PNG")

            # plt.imshow(new_im)
            # new_size = (original_image.size[0] * 2, original_image.size[1])
            # new_im = Image.new('L', new_size)
            # new_im.paste(original_image, (0, 0))
            # pred_image = Image.fromarray(get_picture_array(X_train, index))
            # # pred_image.save(folder_path + 'noisyInput' + str(index) + '.png', format="PNG")
            # new_im.paste(pred_image, (original_image.size[0], 0))
            # new_im.save(folder_path+'origin_VS_noise-'+str(index)+'.png', format="PNG")
            # plt.imshow(new_im)

        return cnn

    def createSAE(input_height, input_width, X_train, X_out):
        encode_size = 200

        cnn1 = NeuralNet(layers=[
            ('input', layers.InputLayer),
            ('hidden', layers.DenseLayer),
            ('hiddenOut', layers.DenseLayer),
            ('output_layer', ReshapeLayer),
        ],

            input_shape=(None, 1, input_width, input_height),
            hidden_num_units= 10000,
            hiddenOut_num_units= 42000,
            output_layer_shape = (([0], -1)),

            update_learning_rate=learning_rate,
            update_momentum=update_momentum,
            update=nesterov_momentum,
            train_split=TrainSplit(eval_size=train_valid_split),
            # batch_iterator_train=BatchIterator(batch_size=batch_size),
            batch_iterator_train=FlipBatchIterator(batch_size=batch_size),
            regression=True,
            max_epochs=epochs,
            verbose=1,
            hiddenLayer_to_output=-3)

        cnn1.fit(X_train, X_out)
        trian_last_hiddenLayer = cnn1.output_hiddenLayer(X_train)
        test_last_hiddenLayer = cnn1.output_hiddenLayer(test_x)

        cnn2 = NeuralNet(layers=[
            ('input', layers.InputLayer),
            ('hidden', layers.DenseLayer),
            ('output_layer', layers.DenseLayer),
        ],

            input_shape=(None,10000),
            hidden_num_units= 3000,
            output_layer_num_units = 10000,

            update_learning_rate=learning_rate,
            update_momentum=update_momentum,
            update=nesterov_momentum,
            train_split=TrainSplit(eval_size=train_valid_split),
            batch_iterator_train=BatchIterator(batch_size=batch_size),
            # batch_iterator_train=FlipBatchIterator(batch_size=batch_size),
            regression=True,
            max_epochs=epochs,
            verbose=1,
            hiddenLayer_to_output=-2)

        trian_last_hiddenLayer = trian_last_hiddenLayer.astype(np.float32)

        cnn2.fit(trian_last_hiddenLayer, trian_last_hiddenLayer)
        trian_last_hiddenLayer = cnn2.output_hiddenLayer(trian_last_hiddenLayer)
        test_last_hiddenLayer = cnn2.output_hiddenLayer(test_last_hiddenLayer)

        cnn3 = NeuralNet(layers=[
            ('input', layers.InputLayer),
            ('hidden', layers.DenseLayer),
            ('output_layer', layers.DenseLayer),
        ],

            input_shape=(None,3000),
            hidden_num_units= 1000,
            output_layer_num_units = 3000,

            update_learning_rate=learning_rate,
            update_momentum=update_momentum,
            update=nesterov_momentum,
            train_split=TrainSplit(eval_size=train_valid_split),
            batch_iterator_train=BatchIterator(batch_size=batch_size),
            # batch_iterator_train=FlipBatchIterator(batch_size=batch_size),
            regression=True,
            max_epochs=epochs,
            verbose=1,
            hiddenLayer_to_output=-2)

        trian_last_hiddenLayer = trian_last_hiddenLayer.astype(np.float32)
        cnn3.fit(trian_last_hiddenLayer, trian_last_hiddenLayer)
        trian_last_hiddenLayer = cnn3.output_hiddenLayer(trian_last_hiddenLayer)
        test_last_hiddenLayer = cnn3.output_hiddenLayer(test_last_hiddenLayer)

        cnn4 = NeuralNet(layers=[
            ('input', layers.InputLayer),
            ('hidden', layers.DenseLayer),
            ('output_layer', layers.DenseLayer),
        ],

            input_shape=(None,1000),
            hidden_num_units= 300,
            output_layer_num_units = 1000,

            update_learning_rate=learning_rate,
            update_momentum=update_momentum,
            update=nesterov_momentum,
            train_split=TrainSplit(eval_size=train_valid_split),
            batch_iterator_train=BatchIterator(batch_size=batch_size),
            # batch_iterator_train=FlipBatchIterator(batch_size=batch_size),
            regression=True,
            max_epochs=epochs,
            verbose=1,
            hiddenLayer_to_output=-2)

        trian_last_hiddenLayer = trian_last_hiddenLayer.astype(np.float32)
        cnn4.fit(trian_last_hiddenLayer, trian_last_hiddenLayer)
        trian_last_hiddenLayer = cnn4.output_hiddenLayer(trian_last_hiddenLayer)
        test_last_hiddenLayer = cnn4.output_hiddenLayer(test_last_hiddenLayer)


        input_layer = cnn1.get_all_layers()[0]
        hidden1_layer = cnn1.get_all_layers()[1]
        hidden1_layer.input_layer = input_layer
        hidden2_layer = cnn2.get_all_layers()[1]
        hidden2_layer.input_layer = hidden1_layer
        hidden3_layer = cnn3.get_all_layers()[1]
        hidden3_layer.input_layer = hidden2_layer
        final_layer = cnn4.get_all_layers()[1]
        final_layer.input_layer = hidden3_layer

        #         out_train = final_layer.get_output(x_train).eval()
        #         out_test = final_layer.get_output(test_x).eval()

        f = gzip.open(folder_path + "output.pkl.gz",'wb')
        cPickle.dump((trian_last_hiddenLayer, test_last_hiddenLayer), f, protocol=2)
        f.close()
        #         f = gzip.open("pickled_images/tmp.pkl.gz", 'rb')
        #         trian_last_hiddenLayer, test_last_hiddenLayer = cPickle.load(f)
        #         f.close()

        return cnn1

    def createCnn_AE(input_height, input_width):
        if categories==20:
            outputLayerSize=20
        else:
            outputLayerSize=15

        encode_size = 1024
        border_mode = "same"

        cnn = NeuralNet(layers=[
            ('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),
            ('conv2', layers.Conv2DLayer),
            ('pool2', layers.MaxPool2DLayer),
            ('conv3', layers.Conv2DLayer),
            ('pool3', layers.MaxPool2DLayer),
            # ('conv4', layers.Conv2DLayer),
            # ('pool4', layers.MaxPool2DLayer),
            ('flatten', ReshapeLayer),  # output_dense
            ('encode_layer', layers.DenseLayer),
            ('hidden', layers.DenseLayer),  # output_dense
            ('unflatten', ReshapeLayer),
            # ('unpool4', Unpool2DLayer),
            # ('deconv4', layers.Conv2DLayer),
            ('unpool3', Unpool2DLayer),
            ('deconv3', layers.Conv2DLayer),
            ('unpool2', Unpool2DLayer),
            ('deconv2', layers.Conv2DLayer),
            ('unpool1', Unpool2DLayer),
            ('deconv1', layers.Conv2DLayer),
            ('output_layer', ReshapeLayer),

            # ('hidden5', layers.DenseLayer),
            # ('hidden6', layers.DenseLayer),
            # ('hidden7', layers.DenseLayer),
            # ('output', layers.DenseLayer)
        ],

            input_shape=(None, 1, input_width, input_height),
            # Layer current size - 1x300x140
            conv1_num_filters=layers_size[0], conv1_filter_size=(5, 5), conv1_border_mode="valid", conv1_nonlinearity=None,
            #Layer current size - NFx296x136
            pool1_pool_size=(2, 2),
            # Layer current size - NFx148x68
            conv2_num_filters=layers_size[1], conv2_filter_size=(5, 5), conv2_border_mode=border_mode, conv2_nonlinearity=None,
            # Layer current size - NFx148x68
            pool2_pool_size=(2, 2),
            # Layer current size - NFx74x34
            conv3_num_filters=layers_size[2], conv3_filter_size=(3, 3), conv3_border_mode=border_mode, conv3_nonlinearity=None,
            # Layer current size - NFx74x34
            pool3_pool_size=(2, 2),

            # conv4_num_filters=layers_size[3], conv4_filter_size=(5, 5), conv4_border_mode=border_mode, conv4_nonlinearity=None,
            # pool4_pool_size=(2, 2),

            # Layer current size - NFx37x17
            flatten_shape=(([0], -1)), # not sure if necessary?
            # Layer current size - NF*37*17
            encode_layer_num_units = encode_size,
            # Layer current size - 200
            hidden_num_units=layers_size[-1] * 37 * 17,
            # Layer current size - NF*37*17
            unflatten_shape=(([0], layers_size[-1], 37, 17)),

            # deconv4_num_filters=layers_size[3], deconv4_filter_size=(5, 5), deconv4_border_mode=border_mode, deconv4_nonlinearity=None,
            # unpool4_ds=(2, 2),

            # Layer current size - NFx37x17
            unpool3_ds=(2, 2),
            # Layer current size - NFx74x34
            deconv3_num_filters=layers_size[-2], deconv3_filter_size=(3, 3), deconv3_border_mode=border_mode, deconv3_nonlinearity=None,
            # Layer current size - NFx74x34
            unpool2_ds=(2, 2),
            # Layer current size - NFx148x68
            deconv2_num_filters=layers_size[-3], deconv2_filter_size=(5, 5), deconv2_border_mode=border_mode, deconv2_nonlinearity=None,
            # Layer current size - NFx148x68
            unpool1_ds=(2, 2),
            # Layer current size - NFx296x136
            deconv1_num_filters=1, deconv1_filter_size=(5, 5), deconv1_border_mode="full", deconv1_nonlinearity=None,
            # Layer current size - 1x300x140
            output_layer_shape = (([0], -1)),
            # Layer current size - 300*140

            # output_num_units=outputLayerSize, output_nonlinearity=None,
            update_learning_rate=learning_rate,
            update_momentum=update_momentum,
            update=nesterov_momentum,
            train_split=TrainSplit(eval_size=train_valid_split),
            # batch_iterator_train=BatchIterator(batch_size=batch_size),
            batch_iterator_train=FlipBatchIterator(batch_size=batch_size),
            regression=True,
            max_epochs=epochs,
            verbose=1,
            hiddenLayer_to_output=-10)
        # on_training_finished=last_hidden_layer,
        return cnn

    def createNNwithDecay(input_height, input_width):
        if categories==20:
            outputLayerSize=20
        else:
            outputLayerSize=15

        cnn = NeuralNet(layers=[
            ('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),
            ('conv2', layers.Conv2DLayer),
            ('pool2', layers.MaxPool2DLayer),
            ('conv3', layers.Conv2DLayer),
            ('pool3', layers.MaxPool2DLayer),
            ('conv4', layers.Conv2DLayer),
            ('pool4', layers.MaxPool2DLayer),
            ('hidden5', layers.DenseLayer),
            ('hidden6', layers.DenseLayer),
            ('hidden7', layers.DenseLayer),
            ('output', layers.DenseLayer)],
            input_shape=(None, 1, input_width, input_height),
            conv1_num_filters=layers_size[0], conv1_filter_size=(5, 5), pool1_pool_size=(2, 2),
            conv2_num_filters=layers_size[1], conv2_filter_size=(9, 9), pool2_pool_size=(2, 2),
            conv3_num_filters=layers_size[2], conv3_filter_size=(11, 11), pool3_pool_size=(4, 2),
            conv4_num_filters=layers_size[3], conv4_filter_size=(8, 5), pool4_pool_size=(2, 2),
            hidden5_num_units=500, hidden6_num_units=200, hidden7_num_units=100,
            output_num_units=outputLayerSize, output_nonlinearity=None,
            update_learning_rate=learning_rate,
            update_rho=update_rho,
            update=rmsprop,
            train_split=TrainSplit(eval_size=train_valid_split),
            batch_iterator_train=BatchIterator(batch_size=batch_size),
            regression=True,
            max_epochs=epochs,
            verbose=1,
            hiddenLayer_to_output=-2)
        #         on_training_finished=last_hidden_layer,
        return cnn

    def last_hidden_layer(s, h):

        print (s.output_last_hidden_layer_(train_x))

    #         input_layer = s.get_all_layers()[0]
    #         last_h_layer = s.get_all_layers()[-2]
    #         f = theano.function(s.X_inputs, last_h_layer.get_output(last_h_layer),allow_input_downcast=True)

    #         myFunc = theano.function(
    #                     inputs=s.input_X,
    #                     outputs=s.h_predict,
    #                     allow_input_downcast=True,
    #                     )
    #         print s.output_last_hidden_layer_(train_x,-2)

    def writeOutputFile(output_file, train_history, layer_info):
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
        output_file.write("Number of images: " + str(end_index) + "\n")
        results_file.write(str(end_index) + "\t")
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
        output_file.write("filters_info:\n" + str(filter_1) + "\n")
        output_file.write(str(filter_2) + "\n")
        output_file.write(str(filter_3) + "\n")
        output_file.write(str(filter_4) + "\n")
        output_file.write(str(filter_5) + "\n")
        output_file.write(str(filter_6) + "\n\n")
        results_file.write("{" + str((filter_1, filter_2, filter_3, filter_4, filter_5, filter_6)) + "]\t")
        output_file.write("Run time[minutes] is: " + str(run_time) + "\n")

        output_file.flush()
        results_file.write(str(time.ctime()) + "\t")
        results_file.write(folder_name + "\n")
        results_file.flush()

    def outputLastLayer_CNN(cnn, X, y=None, test_x=None, test_y=None):
        print ("outputing last hidden layer")  # train_last_hiddenLayer = cnn.output_hiddenLayer(train_x)
        quarter_x = np.floor(X.shape[0] / 4)

        train_last_hiddenLayer1 = cnn.output_hiddenLayer(
            (np.random.binomial(1, 1 - dropout_percent, size=X[:quarter_x].shape) * X[:quarter_x]).astype(
                np.float32).reshape((-1, 1, input_width, input_height)))
        pickle.dump(train_last_hiddenLayer1, open(folder_path + 'encode1.pkl', 'w'))

        print ("after first quarter train output")
        train_last_hiddenLayer2 = cnn.output_hiddenLayer(
            (np.random.binomial(1, 1 - dropout_percent, size=X[quarter_x:2 * quarter_x].shape) * X[quarter_x:2 * quarter_x]).astype(
                np.float32).reshape((-1, 1, input_width, input_height)))
        pickle.dump(train_last_hiddenLayer2, open(folder_path + 'encode2.pkl', 'w'))

        print ("after seconed quarter train output")
        train_last_hiddenLayer3 = cnn.output_hiddenLayer(
            (np.random.binomial(1, 1 - dropout_percent, size=X[2 * quarter_x: 3 * quarter_x].shape) * X[2 * quarter_x:3 * quarter_x]).astype(
                np.float32).reshape((-1, 1, input_width, input_height)))
        pickle.dump(train_last_hiddenLayer3, open(folder_path + 'encode3.pkl', 'w'))

        print ("after third quarter train output")
        train_last_hiddenLayer4 = cnn.output_hiddenLayer(
            (np.random.binomial(1, 1 - dropout_percent, size=X[3 * quarter_x:].shape) * X[3 * quarter_x:]).astype(
                np.float32).reshape((-1, 1, input_width, input_height)))
        pickle.dump(train_last_hiddenLayer4, open(folder_path + 'encode4.pkl', 'w'))

        print ("after all train output")
        if test_x is not None:
            test_last_hiddenLayer = cnn.output_hiddenLayer(test_x)
            print ("after test output")  # lastLayerOutputs = (train_last_hiddenLayer,train_y,test_last_hiddenLayer,test_y)

        # lastLayerOutputs = np.concatenate((train_last_hiddenLayer1, train_last_hiddenLayer2, train_last_hiddenLayer3, train_last_hiddenLayer4), axis=0), y, test_last_hiddenLayer, test_y
        return np.concatenate((train_last_hiddenLayer1, train_last_hiddenLayer2, train_last_hiddenLayer3, train_last_hiddenLayer4), axis=0)

    def outputLastLayer_DAE(train_x, train_y, test_x, test_y):

        # building the SDA
        sDA = StackedDA(layers_size)

        # pre-trainning the SDA
        sDA.pre_train(train_x, noise_rate=input_noise_rate, epochs=pre_train_epochs,LOG=log_file)

        # saving a PNG representation of the first layer
        W = sDA.Layers[0].W.T[:, 1:]
        #         import rl_dae.utils
        #         utils.saveTiles(W, img_shape= (28,28), tile_shape=(10,10), filename="results/res_dA.png")

        # adding the final layer
        #         sDA.finalLayer(train_x, train_y, epochs=softmax_train_epochs)

        # trainning the whole network
        #         sDA.fine_tune(train_x, train_x, epochs=fine_tune_epochs)

        # predicting using the SDA
        testRepresentation = sDA.predict(test_x)
        pred = testRepresentation.argmax(1)

        # let's see how the network did
        #         test_category = test_y.argmax(1)
        e = 0.0
        t = 0.0
        for i in range(test_y.shape[0]):
            if any(test_y[i]):
                e += (test_y[i,pred[i]]==1)
                t += 1

        # printing the result, this structure should result in 80% accuracy
        print ("DAE accuracy: %2.2f%%" % (100 * e / t))
        output_file.write("DAE predict rate:  "+str(100*e/t) + "%\n")

        lastLayerOutputs = (sDA.predict(train_x), train_y, testRepresentation, test_y)
        return lastLayerOutputs #sDA

    start_time = time.clock()
    print ("Start time: ", time.ctime())

    if loadedData is None:
        train_x, train_y, test_x, test_y, svm_data, svm_label = load2d(categories, output_file, input_width, input_height, end_index, multiple_positives, dropout_percent)  # load 2-d data
    else:
        data, svm_data, svm_label = loadedData
        train_x, train_y, test_x, test_y = data

    if zero_meaning:
        train_x = train_x.astype(np.float64)
        mu, sigma = np.mean(train_x.flatten()), np.std(train_x.flatten())
        print("Mean- ", mu)
        print("Std- ", sigma)
        train_x = (train_x - mu) / sigma

    x_train = train_x[:end_index].astype(np.float32).reshape((-1, 1, input_width, input_height))
    x_out = x_train.reshape((x_train.shape[0], -1))
    # test_x = test_x.astype(np.float32).reshape((-1, 1, input_width, input_height))

    cnn = createCSAE(input_height, input_width, x_train, x_out)


    ''' Denoising Autoencoder
    dae = DenoisingAutoencoder(n_hidden=10)
    dae.fit(train_x)
    new_X = dae.transform(train_x)
    print new_X
    '''

    '''Conv Stacked AE
    train_x = np.rint(train_x * 256).astype(np.int).reshape((-1, 1, input_width, input_height ))  # convert to (0,255) int range (we'll do our own scaling)
    mu, sigma = np.mean(train_x.flatten()), np.std(train_x.flatten())

    x_train = train_x.astype(np.float64)
    x_train = (x_train - mu) / sigma
    x_train = x_train.astype(np.float32)

    # we need our target to be 1 dimensional
    x_out = x_train.reshape((x_train.shape[0], -1))

    test_x = np.rint(test_x * 256).astype(np.int).reshape((-1, 1, input_width, input_height ))  # convert to (0,255) int range (we'll do our own scaling)
    # mu, sigma = np.mean(test_x.flatten()), np.std(test_x.flatten())
    test_x = train_x.astype(np.float64)
    test_x = (x_train - mu) / sigma
    test_x = x_train.astype(np.float32)
    '''

    ''' CNN with lasagne
    cnn = createNNwithMomentom(input_height, input_width) if update_rho == None else createNNwithDecay(input_height, input_width)
    cnn.fit(train_x, train_y)
    lastLayerOutputs = outputLastLayer_CNN(cnn, train_x, train_y, test_x, test_y)
    '''

    '''  AE (not Stacked) with Convolutional layers
    cnn = createCnn_AE(input_height, input_width)
    cnn.fit(x_train, x_out)
    '''

    ''' Stacaked AE with lasagne
    cnn = createSAE(input_height, input_width, x_train, x_out)
    '''

    run_time = (time.clock() - start_time) / 60.

    writeOutputFile(output_file, cnn.train_history_, PrintLayerInfo._get_layer_info_plain(cnn))

    print ("Learning took (min)- ", run_time)


    # train_x = np.random.binomial(1, 1 - dropout_percent, size=train_x.shape) * train_x
    # trian_last_hiddenLayer_1 = cnn.output_hiddenLayer(train_x[:5000])
    # trian_last_hiddenLayer_2 = cnn.output_hiddenLayer(train_x[5000:10000])
    # trian_last_hiddenLayer_3 = cnn.output_hiddenLayer(train_x[10000:])
    # print ("Pickling all encoded images:")
    # try:
    #     trian_last_hiddenLayer = outputLastLayer_CNN(cnn, train_x)
    #
    #     # pickle.dump(trian_last_hiddenLayer_1, open(folder_path + 'encode1.pkl', 'w'))
    #     # pickle.dump(trian_last_hiddenLayer_2, open(folder_path + 'encode2.pkl', 'w'))
    #     # pickle.dump(trian_last_hiddenLayer_3, open(folder_path + 'encode3.pkl', 'w'))
    # except:
    #     print ("Could not save encoded images")
    #
    # print ("Runing SVM:")
    # error_rates = run_svm(trian_last_hiddenLayer)
    # results_file.write(str(error_rates) + "\t" + str(np.average(error_rates)))
    # output_file.write(
    #     "SVM Error rates- " + str(error_rates) + "\n Average error- " + str(np.average(error_rates)) + "\n")
    #
    #
    # sys.setrecursionlimit(10000)
    # pickle.dump(cnn, open(folder_path+'conv_ae.pkl', 'w'))
    # ae = pickle.load(open('mnist/conv_ae.pkl','r'))
    # cnn.save_weights_to(folder_path+'conv_ae.np')

    valid_accuracy = cnn.train_history_[-1]['valid_accuracy']
    if valid_accuracy > 0.05:
        return valid_accuracy

    try:
        print("Running SVM")
        errors, aucs = run_svm(cnn, X_train=svm_data, labels=svm_label)
        print("Errors", errors)
        print("AUC", aucs)
        output_file.write("SVM errors: " + str(errors))
        output_file.write("SVM auc: " + str(aucs))
        results_file.write(str(aucs) + "\n")

        output_file.flush()
        results_file.flush()
    except Exception as e:
        print(e)
        print(e.message)

    return valid_accuracy

'''
    # lastLayerOutputs = outputLastLayer_DAE(train_x, train_y, test_x, test_y)

    X_train_pred = cnn.predict(x_train).reshape(-1, input_height, input_width) * sigma + mu
    X_pred = np.rint(X_train_pred).astype(int)
    X_pred = np.clip(X_pred, a_min = 0, a_max = 255)
    X_pred = X_pred.astype('uint8')
    print X_pred.shape , train_x.shape
    

    ###  show random inputs / outputs side by side
    
    def get_picture_array(X, index):
        array = X[index].reshape(input_height, input_width)
        array = np.clip(array, a_min = 0, a_max = 255)
        return  array.repeat(4, axis = 0).repeat(4, axis = 1).astype(np.uint8())
    
    def get_random_images():
        index = np.random.randint(train_x.shape[0])
        print index
        original_image = Image.fromarray(get_picture_array(train_x, index))
        new_size = (original_image.size[0] * 2, original_image.size[1])
        new_im = Image.new('L', new_size)
        new_im.paste(original_image, (0,0))
        rec_image = Image.fromarray(get_picture_array(X_pred, index))
        new_im.paste(rec_image, (original_image.size[0],0))
        new_im.save(FOLDER_PREFIX+'test.png', format="PNG")
    
    get_random_images()
    IPImage(FOLDER_PREFIX+'test.png')
    
    # <codecell>
    
    ## we find the encode layer from our ae, and use it to define an encoding function
    
    encode_layer_index = -1# map(lambda pair : pair[0], cnn.layers).index('encode_layer')
    encode_layer = cnn.get_all_layers()[encode_layer_index]
    
    def get_output_from_nn(last_layer, X):
        indices = np.arange(128, X.shape[0], 128)
        sys.stdout.flush()
    
        # not splitting into batches can cause a memory error
        X_batches = np.split(X, indices)
        out = []
        for count, X_batch in enumerate(X_batches):
            out.append(last_layer.get_output(X_batch).eval())
            sys.stdout.flush()
        return np.vstack(out)
    
    def encode_input(X):
        return get_output_from_nn(encode_layer, X)
    
    X_encoded = encode_input(x_train)
    
    # <codecell>
    
    next_layer = cnn.get_all_layers()[encode_layer_index + 1]
    final_layer = cnn.get_all_layers()[-1]
    new_layer = layers.InputLayer(shape = (None, encode_layer.num_units))
    
    # N.B after we do this, we won't be able to use the original autoencoder , as the layers are broken up
    next_layer.input_layer = new_layer
    
    def decode_encoded_input(X):
        return get_output_from_nn(final_layer, X)
    
    X_decoded = decode_encoded_input(X_encoded) * sigma + mu
    
    X_decoded = np.rint(X_decoded).astype(int)
    X_decoded = np.clip(X_decoded, a_min = 0, a_max = 255)
    X_decoded  = X_decoded.astype('uint8')
    print X_decoded.shape
    
    ### check it worked :
    
    pic_array = get_picture_array(X_decoded, np.random.randint(len(X_decoded)))
    image = Image.fromarray(pic_array)
    image.save(FOLDER_PREFIX+'test1.png', format="PNG")
    IPImage(FOLDER_PREFIX+'test1.png')
    
    # <codecell>
    
    print "running Category Classifier"  
    log_file.flush()  
#     errorRates, aucScores = runSvm(lastLayerOutputs,15) #HIDDEN_LAYER_OUTPUT_FILE_NAME,15)
    errorRates, aucScores = runLibSvm(lastLayerOutputs,15)
#     errorRates, aucScores = runCrossSvm(lastLayerOutputs,15)
#     errorRates, aucScores = runNNclassifier(lastLayerOutputs,15)

    errorRate = np.average(errorRates)
    aucScore = np.average(aucScores)
    
    outputFile.write("\nClassifiers Total Prediction rate is: "+str(100-errorRate) + "\n\n")
    outputFile.write("Classifiers Error rates are:\n"+str(errorRates) + "\n")
    outputFile.write("\nClassifiers Total AUC Score is: "+str(aucScore) + "\n\n")
    outputFile.write("Classifiers AUC Scores are:\n"+str(aucScores) + "\n")
    outputFile.close()
    
    print "saving last layer outputs"
    log_file.flush()  

#     with open(HIDDEN_LAYER_OUTPUT_FILE_NAME,'wb') as f:
#         pickle.dump(lastLayerOutputs, f, -1)
#         f.close()
    f = gzip.open(HIDDEN_LAYER_OUTPUT_FILE_NAME,'wb')
    cPickle.dump(lastLayerOutputs, f, protocol=2)
    f.close() 
    
#     sys.stdout = old_stdout

    log_file.close()

#     write svm data
#     writeDataToFile(HIDDEN_LAYER_OUTPUT_FILE_NAME,SVM_FILE_NAME)
    
    
    ##############################################
#     train_loss = np.array([i["train_loss"] for i in cnn.train_history_])
#     valid_loss = np.array([i["valid_loss"] for i in cnn.train_history_])
#     pyplot.plot(train_loss, linewidth=3, label="train")
#     pyplot.plot(valid_loss, linewidth=3, label="valid")
#     pyplot.grid()
#     pyplot.legend()
#     pyplot.xlabel("epoch")
#     pyplot.ylabel("loss")
#     pyplot.ylim(1e-3, 1)
#     pyplot.yscale("log")
#     pyplot.savefig(FIG_FILE_NAME)
    
    #################################################
    # def plot_sample(x, y, axis):
    #     img = x.reshape(96, 96)
    #     axis.imshow(img, cmap='gray')
    #     axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=10)
    # 
    # X, _ = load(test=True)
    # y_pred = net1.predict(X)
    # 
    # fig = pyplot.figure(figsize=(6, 6))
    # fig.subplots_adjust(
    #     left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    # 
    # for i in range(16):
    #     ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
    #     plot_sample(X[i], y_pred[i], ax)
    # 
    # pyplot.show()
    
    ########## pickle the network ##########
#     print "pickling"    
# #     with open(PICKLES_NET_FILE_NAME,'wb') as f:
# #         pickle.dump(cnn, f, -1)
# #         f.close()
#     f = gzip.open(PICKLES_NET_FILE_NAME,'wb')
#     cPickle.dump(cnn, f, protocol=2)
#     f.close()
'''


def run_all():
    if platform.dist()[0]:
        isUbuntu = True
        print ("Running in Ubuntu")
    else:
        print ("Running in Windows")

    num_labels = 15
    end_index = 8250
    input_noise_rate = 0.2
    zero_meaning = False
    epochs = 25
    folder_name = "CAE_" + str(end_index) + "_3Conv2Pool9Filters_different3000Batch-"+str(time.time())

    # ac1, ac2, ac3, ac4, ac5, ac6, ac7, ac8 = 1, 1, 1, 1, 1, 1, 1, 1

    for i in range(1, 7, 1):
        print("Run #", i)
        try:
            data, svm_data, svm_label = load2d(batch_index=1, num_labels=num_labels, end_index=end_index, TRAIN_PRECENT=1)
            run(layers_size=[32, 32, 64, 32, 32], epochs=epochs, learning_rate=0.06+0.002*i, update_momentum=0.9,
                dropout_percent=input_noise_rate, loadedData=(data, svm_data, svm_label), folder_name=folder_name, end_index=end_index,
                zero_meaning=zero_meaning, activation=None, last_layer_activation=tanh, filters_type=9)

        except Exception as e:
            print("failed to run- ", i)
            print(e)
            # try:
            #     if np.isfinite(ac3) and i % 3 == 0:
            #         ac3 = run(layers_size=[32, 32, 64, 32, 32], epochs=epochs, learning_rate=0.06 + 0.005 * i, update_momentum=0.9,
            #                   dropout_percent=input_noise_rate, loadedData=data, folder_name=folder_name,
            #                   end_index=end_index,
            #                   zero_meaning=zero_meaning, activation=None, last_layer_activation=tanh, filters_type=7)
            #     else:
            #         ac3 = 1
            # except Exception as e:
            #     print("failed to run- ", i)
            #     print(e)
            # try:
            #     if np.isfinite(ac4) and i % 5 == 1:
            #         ac4 = run(layers_size=[32, 32, 64, 32, 32], epochs=epochs, learning_rate=0.01 * i, update_momentum=0.9,
            #                   dropout_percent=input_noise_rate, loadedData=data, folder_name=folder_name,
            #                   end_index=end_index,
            #                   zero_meaning=zero_meaning, activation=None, last_layer_activation=None, filters_type=5)
            #     # else:
            #     #     ac4 = 1
            # except Exception as e:
            #     print("failed to run- ", i)
            #     print(e)
            # try:
            #     if np.isfinite(ac5) and i % 2 == 0:
            #         ac5 = run(layers_size=[32, 32, 64, 32, 32], epochs=6 + 2*i, learning_rate=0.064 + 0.003 * i, update_momentum=0.9,
            #                   dropout_percent=input_noise_rate, loadedData=(data, svm_data, svm_label), folder_name=folder_name,
            #                   end_index=end_index, batch_size=256,
            #                   zero_meaning=zero_meaning, activation=None, last_layer_activation=tanh, filters_type=11)
            #     else:
            #         ac5 = 1
            # except Exception as e:
            #     print("failed to run- ", i)
            #     print(e)
            # try:
            #     if np.isfinite(ac6) and i % 5 == 2:
            #         ac6 = run(layers_size=[32, 32, 64, 32, 32], epochs=epochs, learning_rate=0.01 * i, update_momentum=0.9,
            #                   dropout_percent=input_noise_rate, loadedData=data, folder_name=folder_name,
            #                   end_index=end_index,
            #                   zero_meaning=zero_meaning, activation=None, last_layer_activation=None, filters_type=7)
            #     # else:
            #     #     ac6 = 1
            # except Exception as e:
            #     print("failed to run- ", i)
            #     print(e)
            # try:
            #     if np.isfinite(ac7):
            #         ac7 = run(layers_size=[32, 32, 64, 32, 32], epochs=epochs, learning_rate=0.01 * i, update_momentum=0.9,
            #                   dropout_percent=input_noise_rate, loadedData=data, folder_name=folder_name,
            #                   end_index=end_index,
            #                   zero_meaning=zero_meaning, activation=None, last_layer_activation=tanh, filters_type=5)
            #     else:
            #         ac7 = 1
            # except Exception as e:
            #     print("failed to run- ", i)
            #     print(e)
            # try:
            #     if np.isfinite(ac8) and i % 2 == 0:
            #         ac8 = run(layers_size=[32, 32, 64, 32, 32], epochs=epochs, learning_rate=0.06 + 0.002 * i, update_momentum=0.9,
            #                   dropout_percent=input_noise_rate, loadedData=data, folder_name=folder_name,
            #                   end_index=end_index,
            #                   zero_meaning=zero_meaning, activation=None, last_layer_activation=None, filters_type=9)
            #     # else:
            #     #     ac8 = 1
            # except Exception as e:
            #     print("failed to run- ", i)
            #     print(e)

            # run4()

if __name__ == "__main__":
    import os
    os.environ["DISPLAY"] = ":99"
    # import pydevd
    # pydevd.settrace('132.71.84.233', port=57869, stdoutToServer=True, stderrToServer=True)
    run_all()
