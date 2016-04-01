import theano
from theano import tensor as T
import lasagne
from lasagne import nonlinearities, layers
from lasagne.objectives import aggregate, categorical_crossentropy
from lasagne.layers.helper import get_output
import cPickle as pickle

import time
import os

# from DatasetMngr import DatasetMngr
from ParseCSV import CSVParser
import NN_8
import NN_9
import NN_10

def run():
    PARAMS_FILE_NAME = "results/"+os.path.split(__file__)[1][:-3]+"_parameters.txt"
    PICKLES_NET_FILE_NAME = "results/"+os.path.split(__file__)[1][:-3]+"_picklesNN.pickle"
    PREDICTION_FILE_NAME = "results/"+os.path.split(__file__)[1][:-3]+"_prediction.txt"
    VALIDATION_FILE_NAME = "results/"+os.path.split(__file__)[1][:-3]+"_validation.txt"
    
    LEARNING_RATE = 0.01
    
    NUM_UNITS_HIDDEN_LAYER = [250]
    
    INPUT_LENGTH = 773
    OUTPUT_SIZE = 2
    
    BATCH_SIZE = 250
    
    NUM_OF_EPOCH = 30
    
    
    
    def get_functions():
    
        input_layer=layers.InputLayer(shape=(BATCH_SIZE, INPUT_LENGTH))
        print "input_layer size: " + str(input_layer.shape[0])+","+ str(input_layer.shape[1])
        layer = input_layer
    
        for layer_num in range(len(NUM_UNITS_HIDDEN_LAYER)):
            print "layer_num-"+str(layer_num)
            layer=layers.DenseLayer(layer,
                                       num_units=NUM_UNITS_HIDDEN_LAYER[layer_num],
                                       W=lasagne.init.Normal(0.01),
                                       nonlinearity=nonlinearities.tanh)
    
    
        output_layer=layers.DenseLayer(layer,
                                       num_units=OUTPUT_SIZE,
                                       nonlinearity=nonlinearities.softmax)
    
    
        network_output=get_output(output_layer)
        expected_output=T.ivector()
    
    
        loss_train=aggregate(categorical_crossentropy(network_output, expected_output), mode='mean')
    
        all_weigths=layers.get_all_params(output_layer)
    
        update_rule=lasagne.updates.nesterov_momentum(loss_train, all_weigths, learning_rate=LEARNING_RATE)
        
        print "input_layer_end size: " + str(input_layer.shape[0])+","+ str(input_layer.shape[1])
        train_function=theano.function(inputs=[input_layer.input_var, expected_output],
                                       outputs=loss_train,
                                       updates=update_rule,
                                       allow_input_downcast=True)
    
        prediction = T.argmax(network_output, axis=1)
        accuracy = T.mean(T.eq(prediction, expected_output), dtype=theano.config.floatX)  # @UndefinedVariable
    
        test_function=theano.function(inputs=[input_layer.input_var, expected_output],
                                      outputs=[loss_train, accuracy, prediction],
                                      allow_input_downcast=True)
        
        output_function=theano.function([input_layer.input_var],get_output(output_layer),
                                      allow_input_downcast=True)
    
        return train_function,test_function,output_function
    
    
    
    #############################################
    start_time = time.clock()
    
    train_function, test_function,output_function=get_functions()
    parser = CSVParser()
    print 'loading data...'
    
    
    # load data
    ## train_set_x, train_set_y = parser.parse_csv('data_csv/debug_train.csv')
    train_set_x, train_set_y = parser.parse_csv('data_csv/train.csv')
    valid_set_x, valid_set_y = parser.parse_csv('data_csv/validate.csv')
    test_set_x, test_set_y = parser.parse_csv('data_csv/test.csv')
#     train_set_x = np.array(train_set_x)
#     train_set_y = np.array(train_set_y)
#     valid_set_x = np.array(valid_set_x)
#     valid_set_y = np.array(valid_set_y)
#     test_set_x = np.array(test_set_x)
#     test_set_y = np.array(test_set_y)
    
    
    print(train_set_x.shape[0], 'train samples')
    print(valid_set_x.shape[0], 'test samples')
        
    # len(x_train_set) == 773
    
    
    batch_size = BATCH_SIZE
    
    validation_file = open(VALIDATION_FILE_NAME, "w")
    # train and test the network
    for i in range(NUM_OF_EPOCH):
    
        print '--------------------------------'
        print 'training: epoch {0}:'.format(i+1)
    
        # training
        num_of_batches = len(train_set_x)/batch_size
        for j in range(num_of_batches):
            batch_slice = slice(j*batch_size, (j+1)*batch_size)
            train_function(train_set_x[batch_slice], train_set_y[batch_slice])
    
#     ##############
#         if (i==0):
#             print "first epcoh"
#             batch_slice = slice(i*batch_size, (i+1)*batch_size)
#             out_p = output_function(valid_set_x[batch_slice])
#             print out_p
#             temp_file = open("results/temp.txt", "w")
#             for k in range(len(out_p)):
#                     temp_file.write(str('%.5f' % out_p[k][1]) + '\n')
#             temp_file.close()
#     ####################
                    
        print 'testing on validation set...'
#         print "output 20 valid"
#         outp = output_function(valid_set_x[0:batch_size])
#         print outp
        # testing
        total_accuracy = 0
        num_of_batches = len(valid_set_x)/batch_size
        for j in range(num_of_batches):
            batch_slice = slice(j*batch_size, (j+1)*batch_size)
            error, accuracy, pred = test_function(valid_set_x[batch_slice], valid_set_y[batch_slice])
            out_p = output_function(valid_set_x[batch_slice])
            if (i==NUM_OF_EPOCH-1):
                for k in range(len(out_p)):
                    validation_file.write(str('%.5f' % out_p[k][1]) + '\n')
            total_accuracy += accuracy
    
    
        # evaluating error
        total_accuracy/=num_of_batches
        print 'error is: {0}'.format(1-total_accuracy)
    
    
    validation_file.close()
    
    print "output 20 valid"
    outp = output_function(valid_set_x[0:BATCH_SIZE])
    print outp
    print "output 20 test"
    out_p = output_function(test_set_x[0:BATCH_SIZE])
    print out_p

    ########## pickle the network ##########
    print "pickling"
    fileObject = open(PICKLES_NET_FILE_NAME,'wb')
    pickle.dump(test_function,fileObject)
    fileObject.close()
    
    # save the network's parameters
    outputFile = open(PARAMS_FILE_NAME, "w")
    outputFile.write("learning rate: " + str(LEARNING_RATE))
    outputFile.write("batch size: " + str(BATCH_SIZE))
    outputFile.write("num epochs: " + str(NUM_OF_EPOCH))
    outputFile.write("num units hidden layers: " + str(NUM_UNITS_HIDDEN_LAYER))
    outputFile.write("error is: "+str(1-total_accuracy))
    
    end_time = time.clock()
    print "time"
    print (end_time - start_time) / 60.
    
    prediction_file = open(PREDICTION_FILE_NAME, "w")
    test_fake_accuracy=0
    for i in range(len(test_set_x)):
        batch_slice = slice(i, i+1)
        error, accuracy, pred = test_function(test_set_x[batch_slice], test_set_y[batch_slice])
        out_p = output_function(test_set_x[batch_slice])
        for j in range(len(out_p)):
            prediction_file.write(str('%.5f' % out_p[j][1]) + '\n')    
        test_fake_accuracy += accuracy
    
    
        # evaluating error
    test_fake_accuracy/=num_of_batches
    print 'error is: {0}'.format(1-test_fake_accuracy)
    
    prediction_file.close()
    
if __name__ == "__main__":
    NN_8.run()
    NN_9.run()
    run()
    NN_10.run()