

from lasagne import layers
from lasagne.updates import nesterov_momentum
from numpy.f2py.auxfuncs import isstring
from sklearn.metrics import roc_auc_score
from nolearn.lasagne import NeuralNet

from logistic_sgd import load_data
from nolearn.lasagne import BatchIterator
from nolearn.lasagne import TrainSplit
import numpy as np
import cPickle as pickle
import gzip



def runNNclassifier(data_pointer=None,FOLDER_NAME="defualt",LEARNING_RATE=0.04, UPDATE_MOMENTUM=0.9, UPDATE_RHO=None, NUM_OF_EPOCH=15,
                    NUM_CAT=15,end_index=16351,input_length=100, TRAIN_VALIDATION_SPLIT=0.2,
                    NUM_UNITS_HIDDEN_LAYER=[50, 10], BATCH_SIZE=40):

    def getNN(pickledFilePath=None):
        if pickledFilePath is not None : #isinstance(pickledFilePath, str):
            try:
                with open(pickledFilePath,'rb') as f:
                    classifier_net = pickle.load(f)
                    f.close()
            except :
                f = gzip.open(pickledFilePath, 'rb')
                classifier_net = pickle.load(f)
                f.close()
        else:
            classifier_net = NeuralNet(layers=[
                        ('input', layers.InputLayer), 
                        ('hidden1', layers.DenseLayer), 
                        ('hidden2', layers.DenseLayer), 
                        ('output', layers.DenseLayer)], 
                    input_shape=(None, input_length), 
                    hidden1_num_units=NUM_UNITS_HIDDEN_LAYER[0], hidden2_num_units=NUM_UNITS_HIDDEN_LAYER[1], 
                    output_num_units=1, 
                    output_nonlinearity=None, 
                    update_learning_rate=LEARNING_RATE, 
                    update_momentum=UPDATE_MOMENTUM,
                    update=nesterov_momentum,
                    train_split=TrainSplit(eval_size=TRAIN_VALIDATION_SPLIT), 
                    batch_iterator_train=BatchIterator(batch_size=BATCH_SIZE), 
                    regression=True, 
                    max_epochs=NUM_OF_EPOCH, 
                    verbose=1, 
                    hiddenLayer_to_output=-2)
        
        return classifier_net
    
    def checkLabelPredict(pickledFilePath,labelNumber=0):
        if isinstance(pickledFilePath, str):
            try:
                with open(pickledFilePath,'rb') as f:
                    ob = pickle.load(f)
                    f.close()
            except :
                f = gzip.open(pickledFilePath, 'rb')
                ob = pickle.load(f)
                f.close()
        else:
            ob=pickledFilePath
        train_params, train_labels, test_params, test_labels = ob;
        
#         if external_lables is not None:
#             train_labels = external_lables[:train_params.shape[0]]
#             test_labels = external_lables[train_params.shape[0]:train_params.shape[0]+test_params.shape[0]]
#         
        train_y=np.zeros((train_params.shape[0],1))
        for lb in range(0,train_labels.shape[0]):
            train_y[lb]=train_labels[lb][labelNumber]
        train_y=np.array(train_y,np.float32)
        
        test_y=np.zeros((test_params.shape[0],1))
        for lb in range(0,test_params.shape[0]):
            test_y[lb]=test_labels[lb][labelNumber]
        test_y=np.array(test_y,np.float32)
# 
#         train_params = train_params.reshape(-1, train_params.shape[1])
#         test_params = test_params.reshape(-1,  test_params.shape[1])
#         train_y = np.transpose(train_y) #.reshape(-1, train_y.shape[0])
#         test_y = np.transpose(test_y) #.reshape(-1,  test_y.shape[0])
        
        error_rate=-100
        auc_score=-100
        
        classifier_net = getNN()
                
        classifier_net.fit(train_params, train_y)
        test_predict = classifier_net.predict(test_params)

        differ = test_predict-test_y
        error_rate = np.sum(np.abs(differ))/differ.shape[0] * 100
        auc_score = roc_auc_score(test_y, test_predict)
    
        print differ 
        print "        Error- ", error_rate, "%"
        print "            RocAucScore- ", auc_score       
        return (error_rate, auc_score)
    
    if data_pointer is None:
        data_pointer  = load_data(NUM_CAT)       
    
    errorRates = np.zeros(NUM_CAT)
    aucScores = np.zeros(NUM_CAT)
    for i in range(0,NUM_CAT):
        labelPredRate,labelAucScore = checkLabelPredict(data_pointer,i) 
        errorRates[i] = labelPredRate
        aucScores[i] = labelAucScore
    
    errorRate = np.average(errorRates)
    aucAverageScore = np.average(aucScores)
    print "Average error- ", errorRate, "%"
    print "Prediction rate- ", 100-errorRate, "%"
    print "Average Auc Score- ", aucAverageScore
    
    return (errorRates,aucScores)

if __name__ == "__main__":
    main_dir = "C:\\Users\\Ido\\workspace\\ISH_Lasagne\\src\\DeepLearning\\results\\Best_FixedPositives_0.2\\run_"
    outputPickle = "\\hiddenLayerOutput.pkl.gz"
    
    for i in range(12):
        pickName = main_dir + str(i) + outputPickle
        runNNclassifier(pickName,NUM_CAT=15)
    
    

        