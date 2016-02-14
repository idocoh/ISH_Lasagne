# http://scikit-learn.org/stable/tutorial/basic/tutorial.html
from sklearn import svm
import cPickle as pickle
import numpy as np
from pickleImages import getTopCatVector
from numpy.f2py.auxfuncs import isstring
from sklearn.metrics import roc_auc_score
import gzip
from articleCat_CNN_SVM import load2d
from nolearn.lasagne import NeuralNet


def getNN(pickledFilePath):
    if isinstance(pickledFilePath, str):
        try:
            with open(pickledFilePath,'rb') as f:
                net = pickle.load(f)
                f.close()
        except :
            f = gzip.open(pickledFilePath, 'rb')
            net = pickle.load(f)
            f.close()
    else:
        net = pickledFilePath
    
    return net

def getError(labelNumber,test_predict,test_y):
    
    differ = test_predict[:,labelNumber]-test_y[:,labelNumber]
    error_rate = np.sum(np.abs(differ))/differ.shape[0] * 100
    auc_score = roc_auc_score(test_y[:,labelNumber], test_predict[:,labelNumber])
    
    print "label_", labelNumber
    print "    CNN predict Average error- ", error_rate, "%"
    print "    CNN Auc Score- ", auc_score
    
    return (error_rate,auc_score)
              
def runPredecitNN(pickledName,end_index=16351,MULTI_POSITIVES=20,dropout_percent=0.1,num_labels=False):
    
    net = getNN(pickledName)
    
    X, y, test_x, test_y  = load2d(num_labels,end_index=end_index,MULTI_POSITIVES=MULTI_POSITIVES,dropout_percent=dropout_percent)  # load 2-d data

    test_predict = net.predict(test_x)
           
    
    num_labels=test_predict.shape[1]
    errorRates = np.zeros(num_labels)
    aucScores = np.zeros(num_labels)
    for i in range(0,num_labels-1):
        try:
            labelPredRate,labelAucScore = getError(i,test_predict,test_y) 
            errorRates[i] = labelPredRate
            aucScores[i] = labelAucScore
        except ValueError:
            print "No label ", i
    
    errorRate = np.average(errorRates)
    aucAverageScore = np.average(aucScores)
    print "CNN predict Average error- ", errorRate, "%"
    print "CNN Prediction rate- ", 100-errorRate, "%"
    print "CNN Auc Score- ", aucAverageScore
    
    return (errorRate,aucAverageScore)
    
if __name__ == "__main__":
    for i in range(0,6):
        print "run_",i
        pickName = "C:\\Users\\Ido\\workspace\\ISH_Lasagne\\src\\DeepLearning\\results\\running\\run_"+str(i)+"\\picklesNN.pkl.gz"
        runPredecitNN(pickName,end_index=1000,MULTI_POSITIVES=75,dropout_percent=0.3,num_labels=15)
#     pickName = "C:\\Users\\Ido\\workspace\\ISH_Lasagne\\src\\DeepLearning\\results\\articleCat\\run_1.1\\picklesNN.pkl.gz"
#     runPredecitNN(pickName,end_index=0,MULTI_POSITIVES=60,dropout_percent=0.3,num_labels=15)
#     pickName = "C:\\Users\\Ido\\workspace\\ISH_Lasagne\\src\\DeepLearning\\results\\articleCat\\run_1_full\\picklesNN.pkl.gz"
#     runPredecitNN(pickName,USE_TOP_CAT=True)
#     pickName = "C:\\Users\\Ido\\workspace\\ISH_Lasagne\\src\\DeepLearning\\results\\articleCat\\run_2_full\\picklesNN.pkl.gz"
#     runPredecitNN(pickName,USE_TOP_CAT=False)

#     pickName = "C:\\Users\\Ido\\workspace\\ISH_Lasagne\\src\\DeepLearning\\results\\articleCat\\run_0.3-bestFullImagesOneEpoch\\picklesNN.pkl.gz"
#     runPredecitNN(pickName,USE_TOP_CAT=True)

  
