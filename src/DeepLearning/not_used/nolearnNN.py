
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet

from nolearn.lasagne import TrainSplit
from nolearn.lasagne import BatchIterator
from nolearn.lasagne import PrintLayerInfo

from lasagne.objectives import aggregate, squared_error

from logistic_sgd import load_data
from  matplotlib import pyplot
import cPickle as pickle
import numpy as np
import time
import os


counter = 0



def run(LEARNING_RATE=0.04,  UPDATE_MOMENTUM=0.9, NUM_OF_EPOCH=50, OUTPUT_SIZE = 20 , input_width=300, input_height=140,
                    dataset='ISH.pkl.gz', TRAIN_VALIDATION_SPLIT=0.2, #activation=lasagne.nonlinearities.tanh, #rectify
                    NUM_UNITS_HIDDEN_LAYER=[5, 10, 20, 40], BATCH_SIZE=40, toShuffleInput = False , withZeroMeaning = False):
    
    global counter
    FILE_PREFIX =  os.path.split(dataset)[1][4:16] #os.path.split(__file__)[1][:-3]
    PARAMS_FILE_NAME = "results/"+FILE_PREFIX+"_parameters_"+str(counter)+".txt"
    FIG_FILE_NAME = "results/"+FILE_PREFIX+"_fig_"+str(counter)
    PICKLES_NET_FILE_NAME = "results/"+FILE_PREFIX+"_picklesNN_"+str(counter)+".pickle"
#     VALIDATION_FILE_NAME = "results/"+os.path.split(__file__)[1][:-3]+"_validation_"+str(counter)+".txt"
#     PREDICTION_FILE_NAME = "results/"+os.path.split(__file__)[1][:-3]+"_prediction.txt"
    counter +=1

    outputFile = open(PARAMS_FILE_NAME, "w")   
    
    def load2d(dataset='ISH.pkl.gz', toShuffleInput = False , withZeroMeaning = False):
        print 'loading data...'   
    
        datasets = load_data(dataset, toShuffleInput, withZeroMeaning)
    
        train_set_x, train_set_y = datasets[0]
    #     valid_set_x, valid_set_y = datasets[1]
    #     test_set_x, test_set_y = datasets[2]
        
        train_set_x = train_set_x.reshape(-1, 1, input_width, input_height)
        print(train_set_x.shape[0], 'train samples')
        return train_set_x, train_set_y


    def writeOutputFile(outputFile,train_history,layer_info):
        # save the network's parameters
        outputFile.write("error is: "+str(1-train_history[-1]['valid_accuracy']) + "\n")
        outputFile.write("time is: "+str(run_time) + "\n\n")
        
        outputFile.write("learning rate: " + str(LEARNING_RATE) + "\n")
        outputFile.write("momentum: " + str(UPDATE_MOMENTUM) + "\n")
        outputFile.write("batch size: " + str(BATCH_SIZE) + "\n")
        outputFile.write("num epochs: " + str(NUM_OF_EPOCH) + "\n")
        outputFile.write("num units hidden layers: " + str(NUM_UNITS_HIDDEN_LAYER) + "\n")
#         outputFile.write("activation func: " + str(activation) + "\n")
        outputFile.write("train/validation split: " + str(TRAIN_VALIDATION_SPLIT) + "\n")
        outputFile.write("toShuffleInput: " + str(toShuffleInput) + "\n")
        outputFile.write("withZeroMeaning: " + str(withZeroMeaning) + "\n\n")
        
        outputFile.write("history: " + str(train_history) + "\n\n")
        outputFile.write("layer_info:\n" + str(layer_info) + "\n")

    start_time = time.clock()
       
    
    net2 = NeuralNet(
        layers=[
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
            ('output', layers.DenseLayer),
            ],
        input_shape=(None, 1, input_width, input_height),
        conv1_num_filters=NUM_UNITS_HIDDEN_LAYER[0], conv1_filter_size=(5, 5), pool1_pool_size=(2, 2),
        conv2_num_filters=NUM_UNITS_HIDDEN_LAYER[1], conv2_filter_size=(9, 9), pool2_pool_size=(2, 2),
        conv3_num_filters=NUM_UNITS_HIDDEN_LAYER[2], conv3_filter_size=(11, 11), pool3_pool_size=(4, 2),
        conv4_num_filters=NUM_UNITS_HIDDEN_LAYER[3], conv4_filter_size=(8, 5), pool4_pool_size=(2, 2),
        hidden5_num_units=500, hidden6_num_units=200, hidden7_num_units=100,
        output_num_units=20, output_nonlinearity=None,
    
        update_learning_rate=LEARNING_RATE,
        update_momentum=UPDATE_MOMENTUM,
        train_split=TrainSplit(eval_size=TRAIN_VALIDATION_SPLIT),
        batch_iterator_train=BatchIterator(batch_size=BATCH_SIZE),
    
        regression=True,
        max_epochs=NUM_OF_EPOCH,
        verbose=1,
        )

    
    
    
    X, y = load2d()  # load 2-d data
    net2.fit(X, y)
       
    
    run_time = (time.clock() - start_time) / 60.
    
    writeOutputFile(outputFile,net2.train_history_,PrintLayerInfo._get_layer_info_plain(net2))

    # import numpy as np
    # np.sqrt(0.003255) * 48
    
    ##############################################
    train_loss = np.array([i["train_loss"] for i in net2.train_history_])
    valid_loss = np.array([i["valid_loss"] for i in net2.train_history_])
    pyplot.plot(train_loss, linewidth=3, label="train")
    pyplot.plot(valid_loss, linewidth=3, label="valid")
    pyplot.grid()
    pyplot.legend()
    pyplot.xlabel("epoch")
    pyplot.ylabel("loss")
    pyplot.ylim(1e-3, 1)
    pyplot.yscale("log")
    pyplot.savefig(FIG_FILE_NAME)
    
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
    print "pickling"    
    with open(PICKLES_NET_FILE_NAME,'wb') as f:
        pickle.dump(net2, f, -1)
        f.close()
        
    

    
    
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
    
def run_All():
    
    dat='pickled_images/ISH-noLearn_1500_300_140.pkl.gz'
    run(LEARNING_RATE=0.1, NUM_OF_EPOCH=35, NUM_UNITS_HIDDEN_LAYER=[5, 10, 20, 40], BATCH_SIZE=1, toShuffleInput = False , withZeroMeaning = False,dataset=dat)
    run(LEARNING_RATE=0.07, NUM_OF_EPOCH=35, NUM_UNITS_HIDDEN_LAYER=[5, 10, 20, 40], BATCH_SIZE=5, toShuffleInput = False , withZeroMeaning = False,dataset=dat)    
    run(LEARNING_RATE=0.09, NUM_OF_EPOCH=35, NUM_UNITS_HIDDEN_LAYER=[5, 10, 20, 40], BATCH_SIZE=5, toShuffleInput = False , withZeroMeaning = True,dataset=dat)
    run(LEARNING_RATE=0.2, NUM_OF_EPOCH=35, NUM_UNITS_HIDDEN_LAYER=[5, 10, 20, 40], BATCH_SIZE=5, toShuffleInput = False , withZeroMeaning = False,dataset=dat)
    run(LEARNING_RATE=0.04, NUM_OF_EPOCH=35, NUM_UNITS_HIDDEN_LAYER=[5, 10, 20, 40], BATCH_SIZE=5, toShuffleInput = False , withZeroMeaning = True,dataset=dat)

if __name__ == "__main__":
    run_All()