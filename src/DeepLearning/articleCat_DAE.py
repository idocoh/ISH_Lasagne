
import os
import time

from lasagne import layers
from lasagne.objectives import aggregate, squared_error
from lasagne.updates import nesterov_momentum
from lasagne.updates import rmsprop
from  matplotlib import pyplot
import theano
from autoencoder import DenoisingAutoencoder

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


FILE_SEPARATOR="/"
counter = 0
# X = 0

def load2d(num_labels,outputFile=None, input_width=300, input_height=140,end_index=16351,MULTI_POSITIVES=20,dropout_percent=0.1, dataset='ISH.pkl.gz', toShuffleInput = False , withZeroMeaning = False):
    print 'loading data...'   

    datasets = load_data(dataset, toShuffleInput=toShuffleInput, withZeroMeaning=withZeroMeaning,end_index=end_index,MULTI_POSITIVES=MULTI_POSITIVES,dropout_percent=dropout_percent, labelset=num_labels)

    train_set_x, train_set_y = datasets[0]
#     valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    
#     train_set_x = train_set_x.reshape(-1, 1, input_width, input_height)
# #         valid_set_x = valid_set_x.reshape(-1, 1, input_width, input_height)
#     test_set_x = test_set_x.reshape(-1, 1, input_width, input_height)

    print(train_set_x.shape[0], 'train samples')
    if outputFile is not None:
        outputFile.write("Number of training examples: "+str(train_set_x.shape[0]) + "\n\n")
    return train_set_x, train_set_y, test_set_x, test_set_y



def run(loadedData=None,FOLDER_NAME="defualt",LEARNING_RATE=0.04, UPDATE_MOMENTUM=0.9, UPDATE_RHO=None, NUM_OF_EPOCH=15, input_width=300, input_height=140,
                    dataset='withOutDataSet', TRAIN_VALIDATION_SPLIT=0.2, MULTI_POSITIVES=20, dropout_percent=0.1, USE_NUM_CAT=20,end_index=16351, #activation=lasagne.nonlinearities.tanh, #rectify
                    NUM_UNITS_HIDDEN_LAYER=[5, 10, 20, 40], BATCH_SIZE=40, toShuffleInput = False , withZeroMeaning = False,
                    input_noise_rate=0.3,pre_train_epochs=1,softmax_train_epochs=2,fine_tune_epochs=2):
    
    global counter
#     FILE_PREFIX =  os.path.split(dataset)[1][:-6] #os.path.split(__file__)[1][:-3]
    FOLDER_PREFIX = "results_dae"+FILE_SEPARATOR+FOLDER_NAME+FILE_SEPARATOR+"run_"+str(counter)+FILE_SEPARATOR
    if not os.path.exists(FOLDER_PREFIX):
        os.makedirs(FOLDER_PREFIX)
    
    PARAMS_FILE_NAME = FOLDER_PREFIX + "parameters.txt"
    HIDDEN_LAYER_OUTPUT_FILE_NAME = FOLDER_PREFIX +"hiddenLayerOutput.pkl.gz"
    FIG_FILE_NAME = FOLDER_PREFIX + "fig"
    PICKLES_NET_FILE_NAME = FOLDER_PREFIX + "picklesNN.pkl.gz"
    SVM_FILE_NAME = FOLDER_PREFIX + "svmData.txt"
#     VALIDATION_FILE_NAME = "results/"+os.path.split(__file__)[1][:-3]+"_validation_"+str(counter)+".txt"
#     PREDICTION_FILE_NAME = "results/"+os.path.split(__file__)[1][:-3]+"_prediction.txt"
    counter +=1

    outputFile = open(PARAMS_FILE_NAME, "w")   
    
 

    def createNNwithMomentom(input_height, input_width):
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
            update_momentum=UPDATE_MOMENTUM,
            update=nesterov_momentum, 
            train_split=TrainSplit(eval_size=TRAIN_VALIDATION_SPLIT), 
            batch_iterator_train=BatchIterator(batch_size=BATCH_SIZE), 
            regression=True, 
            max_epochs=NUM_OF_EPOCH, 
            verbose=1, 
            hiddenLayer_to_output=-2)
    #         on_training_finished=last_hidden_layer,
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
        
        print s.output_last_hidden_layer_(X)
#         input_layer = s.get_all_layers()[0]
#         last_h_layer = s.get_all_layers()[-2]
#         f = theano.function(s.X_inputs, last_h_layer.get_output(last_h_layer),allow_input_downcast=True)
 
#         myFunc = theano.function(
#                     inputs=s.input_X,
#                     outputs=s.h_predict,
#                     allow_input_downcast=True,
#                     )
#         print s.output_last_hidden_layer_(X,-2)

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
        print "outputing last hidden layer" #     train_last_hiddenLayer = cnn.output_hiddenLayer(X)
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
        print "after test output" #     lastLayerOutputs = (train_last_hiddenLayer,y,test_last_hiddenLayer,test_y)
        lastLayerOutputs = np.concatenate((train_last_hiddenLayer1, train_last_hiddenLayer2, train_last_hiddenLayer3, train_last_hiddenLayer4), axis=0), y, test_last_hiddenLayer, test_y
        return lastLayerOutputs
        
    def outputLastLayer_DAE(train_x, train_y, test_x, test_y):
        
        # building the SDA
        sDA = StackedDA(NUM_UNITS_HIDDEN_LAYER)
    
        # pre-trainning the SDA
        sDA.pre_train(train_x, noise_rate=input_noise_rate, epochs=pre_train_epochs)
    
        # saving a PNG representation of the first layer
        W = sDA.Layers[0].W.T[:, 1:]
#         import rl_dae.utils 
#         utils.saveTiles(W, img_shape= (28,28), tile_shape=(10,10), filename="results/res_dA.png")
    
        # adding the final layer
        sDA.finalLayer(train_x, train_y, epochs=softmax_train_epochs)
    
        # trainning the whole network
        sDA.fine_tune(train_x, train_y, epochs=fine_tune_epochs)
    
        # predicting using the SDA
        testRepresentation = sDA.predict(test_x) 
        pred = testRepresentation.argmax(1)
    
        # let's see how the network did
        test_category = test_y.argmax(1)
        e = 0.0
        for i in range(len(test_category)):
            e += test_category[i]==pred[i]
    
        # printing the result, this structure should result in 80% accuracy
        print "DAE accuracy: %2.2f%%"%(100*e/len(train_y))
        outputFile.write("DAE predict rate:  "+str(100*e/len(train_y)) + "%\n")
    
        lastLayerOutputs = (sDA.predict(train_x), train_y, testRepresentation, test_y)
        return lastLayerOutputs #sDA
   
    start_time = time.clock()
    print "Start time: " , time.ctime()
           
    if loadedData is None:
    
        X, y, test_x, test_y  = load2d(USE_NUM_CAT,outputFile,input_width,input_height,end_index,MULTI_POSITIVES,dropout_percent)  # load 2-d data
    else:
        X, y, test_x, test_y = loadedData 
    

#     print 1
#     dae = DenoisingAutoencoder(n_hidden=10)
#     print 2
#     dae.fit(X)
#     print 3
#     new_X = dae.transform(X)  
#     print new_X    

    lastLayerOutputs = outputLastLayer_DAE(X, y, test_x, test_y)

#     cnn = createNNwithMomentom(input_height, input_width) if UPDATE_RHO == None else createNNwithDecay(input_height, input_width)   
#     cnn.fit(X, y) 
#     
#     writeOutputFile(outputFile,cnn.train_history_,PrintLayerInfo._get_layer_info_plain(cnn))
# 
#     lastLayerOutputs = outputLastLayer_CNN(cnn, X, y, test_x, test_y)

    run_time = (time.clock() - start_time) / 60.    
    print "learning took- ", run_time
    print "running Category Classifier"    
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
#     with open(HIDDEN_LAYER_OUTPUT_FILE_NAME,'wb') as f:
#         pickle.dump(lastLayerOutputs, f, -1)
#         f.close()
    f = gzip.open(HIDDEN_LAYER_OUTPUT_FILE_NAME,'wb')
    cPickle.dump(lastLayerOutputs, f, protocol=2)
    f.close() 

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


    
def run_All():

    if platform.dist()[0]:
#         global FILE_SEPARATOR
#         FILE_SEPARATOR = "\\"
        print "IsUbuntu"
    else :
        print "IsWindows"

    folderName="gpu_5000_15000"

    num_labels=15
    end_index=300
    MULTI_POSITIVES=0
    input_noise_rate=0.3
    withZeroMeaning=True
    data = load2d(num_labels=num_labels, end_index=end_index, MULTI_POSITIVES=MULTI_POSITIVES, dropout_percent=input_noise_rate,withZeroMeaning=withZeroMeaning)
        
    run(NUM_UNITS_HIDDEN_LAYER=[100],input_noise_rate=0.3,pre_train_epochs=1,softmax_train_epochs=1,fine_tune_epochs=1,loadedData=data,FOLDER_NAME=folderName,USE_NUM_CAT=num_labels,MULTI_POSITIVES=MULTI_POSITIVES, dropout_percent=input_noise_rate,withZeroMeaning=withZeroMeaning)    

#     run(NUM_UNITS_HIDDEN_LAYER=[5000,2000],input_noise_rate=0.3,pre_train_epochs=1,softmax_train_epochs=1,fine_tune_epochs=1,loadedData=data,FOLDER_NAME=folderName,USE_NUM_CAT=num_labels,MULTI_POSITIVES=MULTI_POSITIVES, dropout_percent=input_noise_rate,withZeroMeaning=withZeroMeaning)    
#     run(NUM_UNITS_HIDDEN_LAYER=[2000,500,100],input_noise_rate=input_noise_rate,pre_train_epochs=15,softmax_train_epochs=3,fine_tune_epochs=3,loadedData=data,FOLDER_NAME=folderName,USE_NUM_CAT=num_labels,MULTI_POSITIVES=MULTI_POSITIVES, dropout_percent=input_noise_rate,withZeroMeaning=withZeroMeaning)    
    run(NUM_UNITS_HIDDEN_LAYER=[5000,2000,100],input_noise_rate=input_noise_rate,pre_train_epochs=15,softmax_train_epochs=3,fine_tune_epochs=3,loadedData=data,FOLDER_NAME=folderName,USE_NUM_CAT=num_labels,MULTI_POSITIVES=MULTI_POSITIVES, dropout_percent=input_noise_rate,withZeroMeaning=withZeroMeaning)    
    run(NUM_UNITS_HIDDEN_LAYER=[12000,5000,2000,500,100],input_noise_rate=input_noise_rate,pre_train_epochs=5,softmax_train_epochs=2,fine_tune_epochs=2,loadedData=data,FOLDER_NAME=folderName,USE_NUM_CAT=num_labels,MULTI_POSITIVES=MULTI_POSITIVES, dropout_percent=input_noise_rate,withZeroMeaning=withZeroMeaning)    
    run(NUM_UNITS_HIDDEN_LAYER=[20000,8000,2000,500,100],input_noise_rate=input_noise_rate,pre_train_epochs=5,softmax_train_epochs=2,fine_tune_epochs=2,loadedData=data,FOLDER_NAME=folderName,USE_NUM_CAT=num_labels,MULTI_POSITIVES=MULTI_POSITIVES, dropout_percent=input_noise_rate,withZeroMeaning=withZeroMeaning)
#     run(NUM_UNITS_HIDDEN_LAYER=[15000,7000,3000,1500,700,300,100],input_noise_rate=input_noise_rate,pre_train_epochs=1,softmax_train_epochs=2,fine_tune_epochs=2,loadedData=data,FOLDER_NAME=folderName,USE_NUM_CAT=num_labels,MULTI_POSITIVES=MULTI_POSITIVES, dropout_percent=input_noise_rate,withZeroMeaning=withZeroMeaning)    
#     run(NUM_UNITS_HIDDEN_LAYER=[15000,5000,2000,500,100],input_noise_rate=input_noise_rate,pre_train_epochs=2,softmax_train_epochs=2,fine_tune_epochs=2,loadedData=data,FOLDER_NAME=folderName,USE_NUM_CAT=num_labels,MULTI_POSITIVES=MULTI_POSITIVES, dropout_percent=input_noise_rate,withZeroMeaning=withZeroMeaning)    
#     run(NUM_UNITS_HIDDEN_LAYER=[15000,7000,3000,1500,700,300,100],input_noise_rate=input_noise_rate,pre_train_epochs=2,softmax_train_epochs=2,fine_tune_epochs=2,loadedData=data,FOLDER_NAME=folderName,USE_NUM_CAT=num_labels,MULTI_POSITIVES=MULTI_POSITIVES, dropout_percent=input_noise_rate,withZeroMeaning=withZeroMeaning)    


if __name__ == "__main__":
    run_All()