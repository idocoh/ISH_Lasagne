# http://scikit-learn.org/stable/tutorial/basic/tutorial.html
from sklearn import svm
import cPickle as pickle
import numpy as np
from pickleImages import getTopCatVector
from numpy.f2py.auxfuncs import isstring


def checkLabelPredict(pickledFilePath,labelNumber=0,external_lables=None,NUM_SPLITS = 5):
    if isstring(pickledFilePath):
        with open(pickledFilePath) as f:
            ob = pickle.load(f)
            f.close()
    else:
        ob=pickledFilePath
    train_params, train_y, test_params, test_y = ob;
    allParams = np.concatenate((train_params,test_params),axis=0) 

    
    if external_lables is None:
        allLabels = np.concatenate((train_y,test_y),axis=0)         
    else:
        allLabels = external_lables[:train_params.shape[0]+test_params.shape[0]]

    print "Label- ", labelNumber
    
    negative_labels=np.zeros(allLabels.shape[0])
    positive_labels=np.zeros(allLabels.shape[0])
    negative_params=np.zeros(allParams.shape[0])
    positive_params=np.zeros(allParams.shape[0])
    countNegatives = 1
    countPositives = 1
    for example in range(0,allLabels.shape[0]):
        curLabel = allLabels[example][labelNumber]
        if curLabel:
            negative_labels[countNegatives] = curLabel
            negative_params[countNegatives] = allParams[example]
            countNegatives += 1
        else:
            positive_labels[countPositives] = curLabel
            positive_params[countPositives] = allParams[example]
            countPositives += 1
    negative_labels = negative_labels[:countNegatives]
    positive_labels = positive_labels[:countPositives]
    negative_params = negative_params[:countNegatives]
    positive_params = positive_params[:countPositives]
    
    print "Positives-", countPositives
    print "Negatives-", countNegatives
    
    
    errorRates = np.zeros(NUM_SPLITS)
    negativeDataPart = np.floor(negative_labels.shape[0]/NUM_SPLITS)
    positiveDataPart = np.floor(positive_labels.shape[0]/NUM_SPLITS)
    for i in range[0,NUM_SPLITS-1]:
        if i==0:
            train_labels = np.concatenate((negative_labels[negativeDataPart:],positive_labels[positiveDataPart:]),axis=0)
            test_labels = np.concatenate((negative_labels[:negativeDataPart],positive_labels[:positiveDataPart]),axis=0)
            train_params = np.concatenate((negative_params[negativeDataPart:],positive_params[positiveDataPart:]),axis=0)
            test_params = np.concatenate((negative_params[:negativeDataPart],positive_params[:positiveDataPart]),axis=0)
        elif i==NUM_SPLITS-1:
            train_labels = np.concatenate((negative_labels[:NUM_SPLITS*negativeDataPart],positive_labels[:NUM_SPLITS*positiveDataPart]),axis=0)
            test_labels = np.concatenate((negative_labels[NUM_SPLITS*negativeDataPart:],positive_labels[NUM_SPLITS*positiveDataPart:]),axis=0)
            train_params = np.concatenate((negative_params[:NUM_SPLITS*negativeDataPart],positive_params[:NUM_SPLITS*positiveDataPart]),axis=0)
            test_params = np.concatenate((negative_params[NUM_SPLITS*negativeDataPart:],positive_params[NUM_SPLITS*positiveDataPart:]),axis=0)
        else:         
            train_labels = np.concatenate((negative_labels[:i*negativeDataPart],positive_labels[:i*positiveDataPart],negative_labels[(i+1)*negativeDataPart:],positive_labels[(i+1)*positiveDataPart:]),axis=0)
            test_labels = np.concatenate((negative_labels[i*negativeDataPart:(i+1)*negativeDataPart],positive_labels[i*positiveDataPart:(i+1)*positiveDataPart]),axis=0)
            train_params = np.concatenate((negative_params[:i*negativeDataPart],positive_params[:i*positiveDataPart],negative_params[(i+1)*negativeDataPart:],positive_params[(i+1)*positiveDataPart:]),axis=0)
            test_params = np.concatenate((negative_params[i*negativeDataPart:(i+1)*negativeDataPart],positive_params[i*positiveDataPart:(i+1)*positiveDataPart]),axis=0)

        
        clf = svm.SVC()
        clf.fit(train_params, train_labels)
        
        test_predict = clf.predict(test_params);
        
        differ = test_predict-test_labels
        error_rate = np.sum(np.abs(differ))/differ.shape[0] * 100
        
        print "Predict label- ", labelNumber
        print differ 
        print "Error- ", error_rate, "%"
        
        errorRates[i+1] = error_rate
    
    return np.average(errorRates)

   
#     s = pickle.dumps(clf)
#     from sklearn.externals import joblib
#     joblib.dump(clf, 'filename.pkl') 
#     clf = joblib.load('filename.pkl')
 
def readNewLables(lablesPath,start_index=0,end_index=5000,TRAIN_DATA_PRECENT=0.8,VALIDATION_DATA_PRECENT=0.8):
    y = getTopCatVector(lablesPath,start_index,end_index);  
    
#     with open('topCat.pkl.gz','wb') as f:
#         pickle.dump(y, f, -1)
#         f.close()
    
    dataAmount = end_index-start_index
    train_index = np.floor(dataAmount*TRAIN_DATA_PRECENT);
    validation_index = np.floor(dataAmount*VALIDATION_DATA_PRECENT)
    test_index = dataAmount
    # Divided dataset into 3 parts. 
    train_set_y = y[:train_index]
    val_set_y = y[train_index:validation_index]
    test_set_y = y[validation_index:]   
    
    return (train_set_y,test_set_y)   

def loadNewLabels(pcikledFilePath):
    with open(pcikledFilePath) as f:
        y = pickle.load(f)
        f.close()
    return y
          
def runSvm(pickName,num_labels = 20):
    
#     labelDir = "C:\\Users\\Ido\\workspace\\ISH_Lasagne\\articleCatagorise\\*articleCatagorise.txt"
#     labelDir = "C:\\Users\\Ido\\Pictures\\BrainISHimages\\*TopCat.txt"
#     y = readNewLables(labelDir,end_index=16351)

    y = {
          15: loadNewLabels("C:\\Users\\Ido\\workspace\\ISH_Lasagne\\src\\DeepLearning\\pickled_images\\articleCat.pkl.gz"),
          20: loadNewLabels("C:\\Users\\Ido\\workspace\\ISH_Lasagne\\src\\DeepLearning\\pickled_images\\topCat.pkl.gz"),
          164: loadNewLabels("C:\\Users\\Ido\\workspace\\ISH_Lasagne\\src\\DeepLearning\\pickled_images\\all164cat.pkl.gz"),
          2081: loadNewLabels("C:\\Users\\Ido\\workspace\\ISH_Lasagne\\src\\DeepLearning\\pickled_images\\all164cat2081.pkl.gz"),
        }(num_labels)
    
         
    sum_error = 0
    errorRates = np.zeros(num_labels)
    for i in range(0,num_labels):
        labelPredRate = checkLabelPredict(pickName,i,y) 
        sum_error+= labelPredRate
        errorRates[i] = labelPredRate
    
    errorRate = np.average(errorRates)
    print "Average error- ", errorRate, "%"
    print "Prediction rate- ", 100-errorRate, "%"
    
    return errorRates
    
if __name__ == "__main__":
#     pickName = "C:\\Users\\Ido\\workspace\\ISH_Lasagne\\src\\DeepLearning\\results\\noLearn_50_3_hiddenLayerOutput_0.pickle"
    pickName = "C:\\Users\\Ido\\workspace\\ISH_Lasagne\\src\\DeepLearning\\results\\ISH-noLearn_0_5000_300_140\\run_0\\hiddenLayerOutput.pickle"
    runSvm(pickName,15)

  
