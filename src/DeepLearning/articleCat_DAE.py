
import os
import time
import sys

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
from PIL import Image
import matplotlib.pyplot as plt

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
# X = 0

def load2d(num_labels, outputFile=None, input_width=300, input_height=140, end_index=16351, MULTI_POSITIVES=20,
           dropout_percent=0.1, data_set='ISH.pkl.gz', toShuffleInput = False, withZeroMeaning = False):
    print 'loading data...'

    data_sets = load_data(data_set, toShuffleInput=toShuffleInput, withZeroMeaning=withZeroMeaning, end_index=end_index,
                          MULTI_POSITIVES=MULTI_POSITIVES, dropout_percent=dropout_percent, labelset=num_labels)

    train_set_x, train_set_y = data_sets[0]
#     valid_set_x, valid_set_y = data_sets[1]
    test_set_x, test_set_y = data_sets[2]
    
#     train_set_x = train_set_x.reshape(-1, 1, input_width, input_height)
# #         valid_set_x = valid_set_x.reshape(-1, 1, input_width, input_height)
#     test_set_x = test_set_x.reshape(-1, 1, input_width, input_height)

    print(train_set_x.shape[0], 'train samples')
    if outputFile is not None:
        outputFile.write("Number of training examples: "+str(train_set_x.shape[0]) + "\n\n")
    return train_set_x, train_set_y, test_set_x, test_set_y

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

def run(loadedData=None,FOLDER_NAME="defualt",LEARNING_RATE=0.04, UPDATE_MOMENTUM=0.9, UPDATE_RHO=None, NUM_OF_EPOCH=15,
        input_width=300, input_height=140, dataset='withOutDataSet', TRAIN_VALIDATION_SPLIT=0.2, MULTI_POSITIVES=20,
        dropout_percent=0.1, USE_NUM_CAT=20,end_index=16351, #activation=lasagne.nonlinearities.tanh, #rectify
        NUM_UNITS_HIDDEN_LAYER=[5, 10, 20, 40], BATCH_SIZE=40, toShuffleInput = False ,
        withZeroMeaning = False, input_noise_rate=0.3,pre_train_epochs=1,softmax_train_epochs=2,fine_tune_epochs=2):
    
    global counter
#     FILE_PREFIX =  os.path.split(dataset)[1][:-6] #os.path.split(__file__)[1][:-3]
    FOLDER_PREFIX = "results_dae"+FILE_SEPARATOR+FOLDER_NAME+FILE_SEPARATOR+"run_"+str(counter)+FILE_SEPARATOR
    if not os.path.exists(FOLDER_PREFIX):
        os.makedirs(FOLDER_PREFIX)
    
    PARAMS_FILE_NAME = FOLDER_PREFIX + "parameters.txt"
    HIDDEN_LAYER_OUTPUT_FILE_NAME = FOLDER_PREFIX + "hiddenLayerOutput.pkl.gz"
    FIG_FILE_NAME = FOLDER_PREFIX + "fig"
    PICKLES_NET_FILE_NAME = FOLDER_PREFIX + "picklesNN.pkl.gz"
    SVM_FILE_NAME = FOLDER_PREFIX + "svmData.txt"
    LOG_FILE_NAME = FOLDER_PREFIX + "message.log"

    
#     old_stdout = sys.stdout
#     print "less",LOG_FILE_NAME
    log_file = open(LOG_FILE_NAME, "w")
#     sys.stdout = log_file

#     VALIDATION_FILE_NAME = "results/"+os.path.split(__file__)[1][:-3]+"_validation_"+str(counter)+".txt"
#     PREDICTION_FILE_NAME = "results/"+os.path.split(__file__)[1][:-3]+"_prediction.txt"
    counter += 1

    outputFile = open(PARAMS_FILE_NAME, "w")   
    
    def createSAE(input_height, input_width, X_train, X_out):
        
        X_train = np.random.binomial(1, 1-dropout_percent, size=X_train.shape) * X_train
        
        cnn = NeuralNet(layers=[
                ('input', layers.InputLayer), 
                ('conv1', layers.Conv2DLayer),
                # ('pool1', layers.MaxPool2DLayer),
                # ('conv2', layers.Conv2DLayer),
                # ('conv3', layers.Conv2DLayer),
                # ('unpool1', Unpool2DLayer),
                ('conv4', layers.Conv2DLayer),
                ('output_layer', ReshapeLayer),
                ], 
            input_shape=(None, 1, input_width, input_height), 
            # Layer current size - 1x300x140
            conv1_num_filters=NUM_UNITS_HIDDEN_LAYER[0], conv1_filter_size=(7, 7), conv1_border_mode="same", conv1_nonlinearity=None,
            # pool1_pool_size=(2, 2),
            # conv2_num_filters=NUM_UNITS_HIDDEN_LAYER[1], conv2_filter_size=(5, 5), conv2_border_mode="same", conv2_nonlinearity=None,
            # conv3_num_filters=NUM_UNITS_HIDDEN_LAYER[2], conv3_filter_size=(5, 5), conv3_border_mode="same", conv3_nonlinearity=None,
            # unpool1_ds=(2, 2),
            conv4_num_filters=NUM_UNITS_HIDDEN_LAYER[3], conv4_filter_size=(7, 7), conv4_border_mode="same", conv4_nonlinearity=None,
            output_layer_shape=(([0], -1)),

            update_learning_rate=LEARNING_RATE, 
            update_momentum=UPDATE_MOMENTUM,
            update=nesterov_momentum, 
            train_split=TrainSplit(eval_size=TRAIN_VALIDATION_SPLIT), 
            # batch_iterator_train=BatchIterator(batch_size=BATCH_SIZE),
            batch_iterator_train=FlipBatchIterator(batch_size=BATCH_SIZE), 
            regression=True, 
            max_epochs=NUM_OF_EPOCH, 
            verbose=1,
            hiddenLayer_to_output=-2)
        
        cnn.fit(X_train, X_out)

        # pickle.dump(cnn, open(FOLDER_PREFIX + 'conv_ae.pkl', 'w'))
        # cnn = pickle.load(open(FOLDER_PREFIX + 'conv_ae.pkl','r'))
        # cnn.save_weights_to(FOLDER_PREFIX + 'conv_ae.np')

        X_train_pred = cnn.predict(X_train).reshape(-1, input_height, input_width) * sigma + mu
        X_pred = np.rint(X_train_pred).astype(int)
        X_pred = np.clip(X_pred, a_min=0, a_max=255)
        X_pred = X_pred.astype('uint8')
        print X_pred.shape, train_x.shape
        for i in range(10):
            index = np.random.randint(train_x.shape[0])
            print index

            def get_picture_array(X, index):
                array = X[index].reshape(input_height, input_width)
                array = np.clip(array, a_min=0, a_max=255)
                return array.repeat(4, axis=0).repeat(4, axis=1).astype(np.uint8())
            
            original_image = Image.fromarray(get_picture_array(X_out, index))
            original_image.save(FOLDER_PREFIX + 'original' + str(index) + '.png', format="PNG")

            new_size = (original_image.size[0] * 2, original_image.size[1])
            new_im = Image.new('L', new_size)
            new_im.paste(original_image, (0, 0))
            rec_image = Image.fromarray(get_picture_array(X_pred, index))
            rec_image.save(FOLDER_PREFIX + 'pred' + str(index) + '.png', format="PNG")

            new_im.paste(rec_image, (original_image.size[0], 0))
            new_im.save(FOLDER_PREFIX+'out_VS_pred'+str(index)+'.png', format="PNG")
            plt.imshow(new_im)

            new_size = (original_image.size[0] * 2, original_image.size[1])
            new_im = Image.new('L', new_size)
            new_im.paste(original_image, (0, 0))
            rec_image = Image.fromarray(get_picture_array(X_train, index))
            rec_image.save(FOLDER_PREFIX + 'noisyInput' + str(index) + '.png', format="PNG")

            new_im.paste(rec_image, (original_image.size[0], 0))
            new_im.save(FOLDER_PREFIX+'out_VS_noise'+str(index)+'.png', format="PNG")
            plt.imshow(new_im)

        trian_last_hiddenLayer = cnn.output_hiddenLayer(X_train)
        test_last_hiddenLayer = cnn.output_hiddenLayer(test_x)

        return cnn
        
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

            update_learning_rate=LEARNING_RATE, 
            update_momentum=UPDATE_MOMENTUM,
            update=nesterov_momentum, 
            train_split=TrainSplit(eval_size=TRAIN_VALIDATION_SPLIT), 
            # batch_iterator_train=BatchIterator(batch_size=BATCH_SIZE),
            batch_iterator_train=FlipBatchIterator(batch_size=BATCH_SIZE), 
            regression=True, 
            max_epochs=NUM_OF_EPOCH, 
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

            update_learning_rate=LEARNING_RATE, 
            update_momentum=UPDATE_MOMENTUM,
            update=nesterov_momentum, 
            train_split=TrainSplit(eval_size=TRAIN_VALIDATION_SPLIT), 
            batch_iterator_train=BatchIterator(batch_size=BATCH_SIZE),
            # batch_iterator_train=FlipBatchIterator(batch_size=BATCH_SIZE),
            regression=True, 
            max_epochs=NUM_OF_EPOCH, 
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

            update_learning_rate=LEARNING_RATE, 
            update_momentum=UPDATE_MOMENTUM,
            update=nesterov_momentum, 
            train_split=TrainSplit(eval_size=TRAIN_VALIDATION_SPLIT), 
            batch_iterator_train=BatchIterator(batch_size=BATCH_SIZE),
            # batch_iterator_train=FlipBatchIterator(batch_size=BATCH_SIZE),
            regression=True,
            max_epochs=NUM_OF_EPOCH, 
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

            update_learning_rate=LEARNING_RATE, 
            update_momentum=UPDATE_MOMENTUM,
            update=nesterov_momentum, 
            train_split=TrainSplit(eval_size=TRAIN_VALIDATION_SPLIT), 
            batch_iterator_train=BatchIterator(batch_size=BATCH_SIZE),
            # batch_iterator_train=FlipBatchIterator(batch_size=BATCH_SIZE),
            regression=True, 
            max_epochs=NUM_OF_EPOCH, 
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
        
        f = gzip.open(FOLDER_PREFIX + "output.pkl.gz",'wb')
        cPickle.dump((trian_last_hiddenLayer, test_last_hiddenLayer), f, protocol=2)
        f.close()       
#         f = gzip.open("pickled_images/tmp.pkl.gz", 'rb')
#         trian_last_hiddenLayer, test_last_hiddenLayer = cPickle.load(f)
#         f.close()

        return cnn1       

    def createCnn_AE(input_height, input_width):
        if USE_NUM_CAT==20:
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
            conv1_num_filters=NUM_UNITS_HIDDEN_LAYER[0], conv1_filter_size=(5, 5), conv1_border_mode="valid", conv1_nonlinearity=None,
            #Layer current size - NFx296x136
            pool1_pool_size=(2, 2), 
            # Layer current size - NFx148x68
            conv2_num_filters=NUM_UNITS_HIDDEN_LAYER[1], conv2_filter_size=(5, 5), conv2_border_mode=border_mode, conv2_nonlinearity=None,
            # Layer current size - NFx148x68
            pool2_pool_size=(2, 2), 
            # Layer current size - NFx74x34
            conv3_num_filters=NUM_UNITS_HIDDEN_LAYER[2], conv3_filter_size=(3, 3), conv3_border_mode=border_mode, conv3_nonlinearity=None,
            # Layer current size - NFx74x34
            pool3_pool_size=(2, 2), 

            # conv4_num_filters=NUM_UNITS_HIDDEN_LAYER[3], conv4_filter_size=(5, 5), conv4_border_mode=border_mode, conv4_nonlinearity=None,
            # pool4_pool_size=(2, 2),
 
            # Layer current size - NFx37x17
            flatten_shape=(([0], -1)), # not sure if necessary?
            # Layer current size - NF*37*17
            encode_layer_num_units = encode_size,
            # Layer current size - 200
            hidden_num_units= NUM_UNITS_HIDDEN_LAYER[-1] * 37 * 17,
            # Layer current size - NF*37*17
            unflatten_shape=(([0], NUM_UNITS_HIDDEN_LAYER[-1], 37, 17 )),
            
            # deconv4_num_filters=NUM_UNITS_HIDDEN_LAYER[3], deconv4_filter_size=(5, 5), deconv4_border_mode=border_mode, deconv4_nonlinearity=None,
            # unpool4_ds=(2, 2),

            # Layer current size - NFx37x17
            unpool3_ds=(2, 2),
            # Layer current size - NFx74x34
            deconv3_num_filters=NUM_UNITS_HIDDEN_LAYER[-2], deconv3_filter_size=(3, 3), deconv3_border_mode=border_mode, deconv3_nonlinearity=None,
            # Layer current size - NFx74x34
            unpool2_ds=(2, 2),
            # Layer current size - NFx148x68
            deconv2_num_filters=NUM_UNITS_HIDDEN_LAYER[-3], deconv2_filter_size=(5, 5), deconv2_border_mode=border_mode, deconv2_nonlinearity=None,
            # Layer current size - NFx148x68
            unpool1_ds=(2, 2),  
            # Layer current size - NFx296x136
            deconv1_num_filters=1, deconv1_filter_size=(5, 5), deconv1_border_mode="full", deconv1_nonlinearity=None,
            # Layer current size - 1x300x140
            output_layer_shape = (([0], -1)),
            # Layer current size - 300*140
            
            # output_num_units=outputLayerSize, output_nonlinearity=None,
            update_learning_rate=LEARNING_RATE, 
            update_momentum=UPDATE_MOMENTUM,
            update=nesterov_momentum, 
            train_split=TrainSplit(eval_size=TRAIN_VALIDATION_SPLIT), 
            # batch_iterator_train=BatchIterator(batch_size=BATCH_SIZE),
            batch_iterator_train=FlipBatchIterator(batch_size=BATCH_SIZE), 
            regression=True, 
            max_epochs=NUM_OF_EPOCH, 
            verbose=1, 
            hiddenLayer_to_output=-10)
            # on_training_finished=last_hidden_layer,
        return cnn
    
    def createNNwithDecay(input_height, input_width):
        if USE_NUM_CAT==20:
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
            conv1_num_filters=NUM_UNITS_HIDDEN_LAYER[0], conv1_filter_size=(5, 5), pool1_pool_size=(2, 2), 
            conv2_num_filters=NUM_UNITS_HIDDEN_LAYER[1], conv2_filter_size=(9, 9), pool2_pool_size=(2, 2), 
            conv3_num_filters=NUM_UNITS_HIDDEN_LAYER[2], conv3_filter_size=(11, 11), pool3_pool_size=(4, 2), 
            conv4_num_filters=NUM_UNITS_HIDDEN_LAYER[3], conv4_filter_size=(8, 5), pool4_pool_size=(2, 2), 
            hidden5_num_units=500, hidden6_num_units=200, hidden7_num_units=100, 
            output_num_units=outputLayerSize, output_nonlinearity=None, 
            update_learning_rate=LEARNING_RATE, 
            update_rho=UPDATE_RHO, 
            update=rmsprop, 
            train_split=TrainSplit(eval_size=TRAIN_VALIDATION_SPLIT), 
            batch_iterator_train=BatchIterator(batch_size=BATCH_SIZE), 
            regression=True, 
            max_epochs=NUM_OF_EPOCH, 
            verbose=1, 
            hiddenLayer_to_output=-2)
    #         on_training_finished=last_hidden_layer,
        return cnn
  
    def last_hidden_layer(s, h):
        
        print s.output_last_hidden_layer_(train_x)
#         input_layer = s.get_all_layers()[0]
#         last_h_layer = s.get_all_layers()[-2]
#         f = theano.function(s.X_inputs, last_h_layer.get_output(last_h_layer),allow_input_downcast=True)
 
#         myFunc = theano.function(
#                     inputs=s.input_X,
#                     outputs=s.h_predict,
#                     allow_input_downcast=True,
#                     )
#         print s.output_last_hidden_layer_(train_x,-2)

    def writeOutputFile(outputFile,train_history,layer_info):
        # save the network's parameters
        outputFile.write("Validation set Prediction rate is: "+str((1-train_history[-1]['valid_accuracy'])*100) + "%\n")
        outputFile.write("Run time[minutes] is: "+str(run_time) + "\n\n")
        
        outputFile.write("Training NN on: " + ("20 Top Categorys\n" if USE_NUM_CAT==20 else "Article Categorys\n"))
        outputFile.write("Learning rate: " + str(LEARNING_RATE) + "\n")
        outputFile.write(("Momentum: " + str(UPDATE_MOMENTUM)+ "\n") if (UPDATE_RHO == None) else ("Decay Factor: " + str(UPDATE_RHO)+ "\n") )
        outputFile.write("Batch size: " + str(BATCH_SIZE) + "\n")
        outputFile.write("Num epochs: " + str(NUM_OF_EPOCH) + "\n")
        outputFile.write("Num units hidden layers: " + str(NUM_UNITS_HIDDEN_LAYER) + "\n\n")
#         outputFile.write("activation func: " + str(activation) + "\n")
        outputFile.write("Multipuly Positives by: " + str(MULTI_POSITIVES) + "\n")
        outputFile.write("New Positives Dropout rate: " + str(dropout_percent) + "\n")
        outputFile.write("Train/validation split: " + str(TRAIN_VALIDATION_SPLIT) + "\n")
        outputFile.write("toShuffleInput: " + str(toShuffleInput) + "\n")
        outputFile.write("withZeroMeaning: " + str(withZeroMeaning) + "\n\n")
        
        outputFile.write("history: " + str(train_history) + "\n\n")
        outputFile.write("layer_info:\n" + str(layer_info) + "\n")
        
        outputFile.flush()

    def outputLastLayer_CNN(cnn, X, y, test_x, test_y):
        print "outputing last hidden layer" #     train_last_hiddenLayer = cnn.output_hiddenLayer(train_x)
        quarter_x = np.floor(X.shape[0] / 4)
        train_last_hiddenLayer1 = cnn.output_hiddenLayer(X[:quarter_x])
        print "after first quarter train output"
        train_last_hiddenLayer2 = cnn.output_hiddenLayer(X[quarter_x:2 * quarter_x])
        print "after seconed quarter train output"
        train_last_hiddenLayer3 = cnn.output_hiddenLayer(X[2 * quarter_x:3 * quarter_x])
        print "after third quarter train output"
        train_last_hiddenLayer4 = cnn.output_hiddenLayer(X[3 * quarter_x:])
        print "after all train output"
        test_last_hiddenLayer = cnn.output_hiddenLayer(test_x)
        print "after test output" #     lastLayerOutputs = (train_last_hiddenLayer,train_y,test_last_hiddenLayer,test_y)
        lastLayerOutputs = np.concatenate((train_last_hiddenLayer1, train_last_hiddenLayer2, train_last_hiddenLayer3, train_last_hiddenLayer4), axis=0), y, test_last_hiddenLayer, test_y
        return lastLayerOutputs
        
    def outputLastLayer_DAE(train_x, train_y, test_x, test_y):
        
        # building the SDA
        sDA = StackedDA(NUM_UNITS_HIDDEN_LAYER)
    
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
        print "DAE accuracy: %2.2f%%"%(100*e/t)
        outputFile.write("DAE predict rate:  "+str(100*e/t) + "%\n")
    
        lastLayerOutputs = (sDA.predict(train_x), train_y, testRepresentation, test_y)
        return lastLayerOutputs #sDA
   
    start_time = time.clock()
    print "Start time: " , time.ctime()
           
    if loadedData is None:
        train_x, train_y, test_x, test_y = load2d(USE_NUM_CAT, outputFile, input_width, input_height, end_index, MULTI_POSITIVES, dropout_percent)  # load 2-d data
    else:
        train_x, train_y, test_x, test_y = loadedData
    
#######
#     dae = DenoisingAutoencoder(n_hidden=10)
#     dae.fit(train_x)
#     new_X = dae.transform(train_x)
#     print new_X    

#######
#     lastLayerOutputs = outputLastLayer_DAE(train_x, train_y, test_x, test_y)

#######
    train_x = np.rint(train_x * 256).astype(np.int).reshape((-1, 1, input_width, input_height ))  # convert to (0,255) int range (we'll do our own scaling)
    mu, sigma = np.mean(train_x.flatten()), np.std(train_x.flatten())

    x_train = train_x.astype(np.float64)
    x_train = (x_train - mu) / sigma
    x_train = x_train.astype(np.float32)
     
    # we need our target to be 1 dimensional
    x_out = x_train.reshape((x_train.shape[0], -1))
# 
    test_x = np.rint(test_x * 256).astype(np.int).reshape((-1, 1, input_width, input_height ))  # convert to (0,255) int range (we'll do our own scaling)
    # mu, sigma = np.mean(test_x.flatten()), np.std(test_x.flatten())
    test_x = train_x.astype(np.float64)
    test_x = (x_train - mu) / sigma
    test_x = x_train.astype(np.float32)

######  CNN with lasagne
#     cnn = createNNwithMomentom(input_height, input_width) if UPDATE_RHO == None else createNNwithDecay(input_height, input_width)
#     cnn.fit(train_x, train_y)

######  AE (not Stacked) with Convolutional layers
#     cnn = createCnn_AE(input_height, input_width)
#     cnn.fit(x_train, x_out)

######  Stacaked AE with lasagne
    cnn = createSAE(input_height, input_width, x_train, x_out)
     
    run_time = (time.clock() - start_time) / 60.    

    writeOutputFile(outputFile, cnn.train_history_, PrintLayerInfo._get_layer_info_plain(cnn))
 
    lastLayerOutputs = []    # outputLastLayer_CNN(cnn, train_x, train_y, test_x, test_y)

    print "learning took (min)- ", run_time
    
    
    # <codecell>

    sys.setrecursionlimit(10000)
    
    pickle.dump(cnn, open(FOLDER_PREFIX+'conv_ae.pkl','w'))
    #ae = pickle.load(open('mnist/conv_ae.pkl','r'))
    cnn.save_weights_to(FOLDER_PREFIX+'conv_ae.np')

    return
    # <codecell>
    
    X_train_pred = cnn.predict(x_train).reshape(-1, input_height, input_width) * sigma + mu
    X_pred = np.rint(X_train_pred).astype(int)
    X_pred = np.clip(X_pred, a_min = 0, a_max = 255)
    X_pred = X_pred.astype('uint8')
    print X_pred.shape , train_x.shape
    
    # <codecell>
    
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

def run_all():
    if platform.dist()[0]:
        print "IsUbuntu"
    else:
        print "IsWindows"
    folder_name = "StackedAE_2"

    num_labels = 15
    end_index = 300
    multi_positives = 0
    input_noise_rate = 0.0
    with_zero_meaning = False
    data = load2d(num_labels=num_labels, end_index=end_index, MULTI_POSITIVES=multi_positives, dropout_percent=input_noise_rate,withZeroMeaning=with_zero_meaning)
        
    run(NUM_UNITS_HIDDEN_LAYER=[32, 32, 32, 1], input_noise_rate=0.3, NUM_OF_EPOCH=5, pre_train_epochs=1, softmax_train_epochs=0, fine_tune_epochs=1, loadedData=data, FOLDER_NAME=folder_name, USE_NUM_CAT=num_labels, MULTI_POSITIVES=multi_positives, dropout_percent=input_noise_rate, withZeroMeaning=with_zero_meaning)
    

if __name__ == "__main__":
    import os
    os.environ["DISPLAY"] = ":99"

    # import pydevd
    # pydevd.settrace('132.71.84.233', port=57869, stdoutToServer=True, stderrToServer=True)
    run_all()
    
#IMPORTANT: I have put in comment line number 625 in /home/ido_local/theano-env/lib/python2.7/site-packages/theano/tensor/nnet/conv.py