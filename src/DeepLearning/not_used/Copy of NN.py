import theano
from theano import tensor as T
import lasagne
from lasagne import nonlinearities, layers
from lasagne.objectives import aggregate, squared_error
from lasagne.layers.helper import get_output
# from lasagne.layers import dnn
import cPickle as pickle

import numpy as np
import time
import os

# from DatasetMngr import DatasetMngr
from logistic_sgd import load_data

counter = 0

def run(LEARNING_RATE=0.04, NUM_OF_EPOCH=50, OUTPUT_SIZE = 20 , input_width=300, input_height=140,
                    dataset='ISH.pkl.gz', activation=lasagne.nonlinearities.tanh, #rectify
                    NUM_UNITS_HIDDEN_LAYER=[5, 10, 20, 40], BATCH_SIZE=40, toShuffleInput = False , withZeroMeaning = False):
    
    global counter
    PARAMS_FILE_NAME = "results/"+os.path.split(__file__)[1][:-3]+"_parameters_"+str(counter)+".txt"
    PICKLES_NET_FILE_NAME = "results/"+os.path.split(__file__)[1][:-3]+"_picklesNN_"+str(counter)+".pickle"
    VALIDATION_FILE_NAME = "results/"+os.path.split(__file__)[1][:-3]+"_validation_"+str(counter)+".txt"
#     PREDICTION_FILE_NAME = "results/"+os.path.split(__file__)[1][:-3]+"_prediction.txt"
    counter +=1

    outputFile = open(PARAMS_FILE_NAME, "w")   
    
    start_time = time.clock()
    
#     parser = CSVParser()
    print 'loading data...'
    

    datasets = load_data(dataset,withZeroMeaning=False)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    
    print(train_set_x.shape[0], 'train samples')
    print(valid_set_x.shape[0], 'test samples')
    
    
            
    def get_functions():
        
        def build_model(output_dim=OUTPUT_SIZE, 
                activation = activation, conv_activation = lasagne.nonlinearities.rectify,
                batch_size=BATCH_SIZE,Conv2DLayer=lasagne.layers.Conv2DLayer, MaxPool2DLayer=lasagne.layers.MaxPool2DLayer):
#             Conv2DLayer = dnn.Conv2DDNNLayer, MaxPool2DLayer = dnn.MaxPool2DDNNLayer):
        
            x = T.matrix('x')
            layer_input = x.reshape((batch_size, 1, input_width, input_height))
            
            # Reshape matrix of rasterized images of shape (batch_size, 300, 140)
            # to a 4D tensor, compatible with our NetConvPoolLayer   
            l_in = lasagne.layers.InputLayer(input_var=layer_input , 
                shape=(batch_size, 1, input_width, input_height)
            )        
            #
            outputFile.write( "input_layer size: " + str(l_in.shape[0])+","+ str(l_in.shape[1]) +","+ str(l_in.shape[2]) +","+ str(l_in.shape[3]) + "\n")
            
            # Construct the first convolutional pooling layer:
            # filtering reduces the image size to (300-5+1 , 140-5+1) = (296, 136)
            # maxpooling reduces this further to (296/2, 136/2) = (148, 68)
            l_conv1 = Conv2DLayer(
                l_in,
                num_filters=NUM_UNITS_HIDDEN_LAYER[0],
                filter_size=(5, 5),
                nonlinearity=conv_activation,
                W=lasagne.init.GlorotUniform(),
            )
            #
            out_s = l_conv1.get_output_shape_for((batch_size, 1, input_width, input_height))
            outputFile.write( "conv1_layer size: " + str(out_s) + "\n")

            l_pool1 = MaxPool2DLayer(l_conv1, pool_size=(2, 2))
            #
            out_s = l_pool1.get_output_shape_for(out_s)
            outputFile.write( "pool1_layer size: " + str(out_s) + "\n")


            # Construct the second convolutional pooling layer
            # filtering reduces the image size to (148-9+1, 68-9+1) = (140, 60)
            # maxpooling reduces this further to (140/2, 60/2) = (70, 30)
            l_conv2 = Conv2DLayer(
                l_pool1,
                num_filters=NUM_UNITS_HIDDEN_LAYER[1],
                filter_size=(9, 9),
                nonlinearity=conv_activation,
                W=lasagne.init.GlorotUniform(),
            )
            #
            out_s = l_conv2.get_output_shape_for(out_s)
            outputFile.write( "conv2_layer size: " + str(out_s) + "\n")
            
            l_pool2 = MaxPool2DLayer(l_conv2, pool_size=(2, 2))
            #
            out_s = l_pool2.get_output_shape_for(out_s)
            outputFile.write( "pool2_layer size: " + str(out_s) + "\n")
        
        
            # Construct the third convolutional pooling layer
            # filtering reduces the image size to (70-11+1, 30-11+1) = (60, 20)
            # maxpooling reduces this further to (60/4, 20/2) = (15, 10)
            l_conv3 = Conv2DLayer(
                l_pool2,
                num_filters=NUM_UNITS_HIDDEN_LAYER[2],
                filter_size=(11, 11),
                nonlinearity=conv_activation,
                W=lasagne.init.GlorotUniform(),
            )
            #
            out_s = l_conv3.get_output_shape_for(out_s)
            outputFile.write( "conv3_layer size: " + str(out_s) + "\n")
            
            l_pool3 = MaxPool2DLayer(l_conv3, pool_size=(4, 2))
            #
            out_s = l_pool3.get_output_shape_for(out_s)
            outputFile.write( "pool3_layer size: " + str(out_s) + "\n")
        
        
            # Construct the forth convolutional pooling layer
            # filtering reduces the image size to (15-8+1, 10-5+1) = (8, 6)
            # maxpooling reduces this further to (8/2, 6/2) = (4, 3)
            l_conv4 = Conv2DLayer(
                l_pool3,
                num_filters=NUM_UNITS_HIDDEN_LAYER[3],
                filter_size=(8, 5),
                nonlinearity=conv_activation,
                W=lasagne.init.GlorotUniform(),
            )
            #
            out_s = l_conv4.get_output_shape_for(out_s)
            outputFile.write( "conv4_layer size: " + str(out_s) + "\n")
            
            l_pool4 = MaxPool2DLayer(l_conv4, pool_size=(2, 2))
            #
            out_s = l_pool4.get_output_shape_for(out_s)
            outputFile.write( "pool4_layer size: " + str(out_s) + "\n")
        
            
            # the HiddenLayer being fully-connected, it operates on 2D matrices of
            # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
            # This will generate a matrix of shape (batch_size, NUM_UNITS_HIDDEN_LAYER[3] * 4 * 3),
            l_hidden1 = lasagne.layers.DenseLayer(
                l_pool4,
                num_units=500,
                nonlinearity=activation,
                W=lasagne.init.GlorotUniform(),
            )
            #
            out_s = l_hidden1.get_output_shape_for(out_s)
            outputFile.write( "hidden1_layer size: " + str(out_s) + "\n")
            
    #         l_hidden1_dropout = lasagne.layers.DropoutLayer(l_hidden1, p=0.5)
            
            l_hidden2 = lasagne.layers.DenseLayer(
                l_hidden1,
                num_units=200,
                nonlinearity=activation,
                W=lasagne.init.GlorotUniform(),
            )
            #
            out_s = l_hidden2.get_output_shape_for(out_s)
            outputFile.write( "hidden2_layer size: " + str(out_s) + "\n")
            
    #         l_hidden2_dropout = lasagne.layers.DropoutLayer(l_hidden2, p=0.5)
            
            l_hidden3 = lasagne.layers.DenseLayer(
                l_hidden2,
                num_units=100,
                nonlinearity=activation,
                W=lasagne.init.GlorotUniform(),
            )
            #
            out_s = l_hidden3.get_output_shape_for(out_s)
            outputFile.write( "hidden3_layer size: " + str(out_s) + "\n")
            
    #         l_hidden3_dropout = lasagne.layers.DropoutLayer(l_hidden3, p=0.5)
        
            l_out = lasagne.layers.DenseLayer(
                l_hidden3,
                num_units=output_dim,
                nonlinearity=activation, #softmax,
                W=lasagne.init.GlorotUniform(),
            )
            #
            outputFile.write( "output_layer size: " + str(l_out.get_output_shape_for(out_s)) + "\n\n")
            
            return x,l_out
        
#         input_layer=layers.InputLayer(shape=(BATCH_SIZE, INPUT_LENGTH))
#         print "input_layer size: " + str(input_layer.shape[0])+","+ str(input_layer.shape[1])
#         layer = input_layer
#     
#         for layer_num in range(len(NUM_UNITS_HIDDEN_LAYER)):
#             print "layer_num-"+str(layer_num)
#             layer=layers.DenseLayer(layer,
#                                        num_units=NUM_UNITS_HIDDEN_LAYER[layer_num],
#                                        W=lasagne.init.Normal(0.01),
#                                        nonlinearity=nonlinearities.tanh)
#     
#     
#         output_layer=layers.DenseLayer(layer,
#                                        num_units=OUTPUT_SIZE,
#                                        nonlinearity=nonlinearities.softmax)
#     
        
        x , output_layer = build_model(output_dim=OUTPUT_SIZE,batch_size=BATCH_SIZE)
            
        network_output=get_output(output_layer)
        expected_output=T.matrix()#ivector()
    
    
        loss_train=aggregate(squared_error(network_output, expected_output), mode='mean')
        valid_eror=aggregate(squared_error(network_output, expected_output) > 0.25, mode='sum')
        valid_eror1=squared_error(network_output, expected_output) > 0.25

#         valid_eror=expected_output.shape[1] - np.sum(np.all(np.equal(network_output, expected_output)))

    
        all_weigths=layers.get_all_params(output_layer)
    
        update_rule=lasagne.updates.nesterov_momentum(loss_train, all_weigths, learning_rate=LEARNING_RATE)
        
        train_function=theano.function(inputs=[x, expected_output],
                                       outputs=loss_train,
                                       updates=update_rule,
                                       allow_input_downcast=True)
    
#         prediction = T.argmax(network_output, axis=1)
#         accuracy = T.mean(T.eq(prediction, expected_output), dtype=theano.config.floatX)  # @UndefinedVariable
    
        validation_function=theano.function(inputs=[x, expected_output],
                                      outputs=[valid_eror, valid_eror1],#, accuracy, prediction],
                                      allow_input_downcast=True)
        
        output_function=theano.function([x],get_output(output_layer),
                                      allow_input_downcast=True)
    
        return train_function,validation_function,output_function
        
        
        
    #############################################    
    
    train_function, valid_function,output_function = get_functions()

    validation_file = open(VALIDATION_FILE_NAME, "w")
    # train and test the network
    for i in range(NUM_OF_EPOCH):
    
        print '--------------------------------'
        print 'training: epoch {0}:'.format(i+1)
    
        # training
        num_of_batches = len(train_set_x)/BATCH_SIZE
        for j in range(num_of_batches):
            batch_slice = slice(j*BATCH_SIZE, (j+1)*BATCH_SIZE)
            train_loss=train_function(train_set_x[batch_slice], train_set_y[batch_slice])
#             print train_loss
               
                    
        print 'testing on validation set...'
        total_accuracy = 0
        num_of_batches = len(valid_set_x)/BATCH_SIZE
        for j in range(num_of_batches):
            batch_slice = slice(j*BATCH_SIZE, (j+1)*BATCH_SIZE)
            accuracy = valid_function(valid_set_x[batch_slice], valid_set_y[batch_slice])
            out_p = output_function(valid_set_x[batch_slice])
            print out_p[:BATCH_SIZE]
            print valid_set_y
            print accuracy
##             print accuracy
##             if (i==NUM_OF_EPOCH-1):
##                 for k in range(len(out_p)):
##                     validation_file.write(str('%.5f' % out_p[k][1]) + '\n')
            total_accuracy += accuracy
    
    
        # evaluating error
        total_accuracy/=num_of_batches
        print '~~~~~~~~~~~~~Validation Error is: {0}'.format(1-total_accuracy)
        validation_file.write('{0}\n'.format(1-total_accuracy))

    
    validation_file.close()
    
#     print "output 20 valid"
#     outp = output_function(valid_set_x[0:BATCH_SIZE])
#     print outp
#     print "output 20 test"
#     out_p = output_function(test_set_x[0:BATCH_SIZE])
#     print out_p
    
    end_time = time.clock()
    print "time"
    print (end_time - start_time) / 60.

    ########## pickle the network ##########
    print "pickling"
    fileObject = open(PICKLES_NET_FILE_NAME,'wb')
    pickle.dump(valid_function,fileObject)
    fileObject.close()
     
    
    # save the network's parameters
    outputFile.write("learning rate: " + str(LEARNING_RATE) + "\n")
    outputFile.write("batch size: " + str(BATCH_SIZE) + "\n")
    outputFile.write("num epochs: " + str(NUM_OF_EPOCH) + "\n")
    outputFile.write("num units hidden layers: " + str(NUM_UNITS_HIDDEN_LAYER) + "\n")
    outputFile.write("activation func: " + str(activation) + "\n")
    outputFile.write("toShuffleInput: " + str(toShuffleInput) + "\n")
    outputFile.write("withZeroMeaning: " + str(withZeroMeaning) + "\n\n")
    outputFile.write("error is: "+str(1-total_accuracy) + "\n")
    outputFile.write("time is: "+str((end_time - start_time) / 60.) + "\n")

    
    
# #     prediction_file = open(PREDICTION_FILE_NAME, "w")
#     test_fake_accuracy=0
#     for i in range(len(test_set_x)):
#         batch_slice = slice(i, i+1)
#         error, accuracy, pred = valid_function(test_set_x[batch_slice], test_set_y[batch_slice])
#         out_p = output_function(test_set_x[batch_slice])
# #         for j in range(len(out_p)):
# #             prediction_file.write(str('%.5f' % out_p[j][1]) + '\n')    
#         test_fake_accuracy += accuracy
#     
#     
#         # evaluating error
#     test_fake_accuracy/=num_of_batches
#     print 'error is: {0}'.format(1-test_fake_accuracy)
    
#     prediction_file.close()
    
if __name__ == "__main__":
    run(LEARNING_RATE=0.1, NUM_OF_EPOCH=2, NUM_UNITS_HIDDEN_LAYER=[1, 1, 1, 1], BATCH_SIZE=3, toShuffleInput = False , withZeroMeaning = False)
    run(LEARNING_RATE=0.04, NUM_OF_EPOCH=2, NUM_UNITS_HIDDEN_LAYER=[1, 1, 1, 1], BATCH_SIZE=40, toShuffleInput = False , withZeroMeaning = False)

    run(LEARNING_RATE=0.1, NUM_OF_EPOCH=25, NUM_UNITS_HIDDEN_LAYER=[5, 10, 20, 40], BATCH_SIZE=40, toShuffleInput = False , withZeroMeaning = False)
    run(LEARNING_RATE=0.04, NUM_OF_EPOCH=25, NUM_UNITS_HIDDEN_LAYER=[5, 10, 20, 40], BATCH_SIZE=40, toShuffleInput = False , withZeroMeaning = False)    
    run(LEARNING_RATE=0.04, NUM_OF_EPOCH=25, NUM_UNITS_HIDDEN_LAYER=[5, 10, 20, 40], BATCH_SIZE=40, toShuffleInput = False , withZeroMeaning = True)
    run(LEARNING_RATE=0.04, NUM_OF_EPOCH=25, NUM_UNITS_HIDDEN_LAYER=[5, 10, 20, 40], BATCH_SIZE=40, toShuffleInput = True , withZeroMeaning = False)
    run(LEARNING_RATE=0.04, NUM_OF_EPOCH=25, NUM_UNITS_HIDDEN_LAYER=[5, 10, 20, 40], BATCH_SIZE=40, toShuffleInput = True , withZeroMeaning = True)

    