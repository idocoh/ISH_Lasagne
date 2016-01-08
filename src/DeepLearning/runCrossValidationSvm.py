# http://scikit-learn.org/stable/tutorial/basic/tutorial.html
from sklearn import svm
import cPickle as pickle
import numpy as np
from pickleImages import getTopCatVector
from numpy.f2py.auxfuncs import isstring
from sklearn.metrics import roc_auc_score



def getData(pickledFilePath, external_lables=None):
    if isinstance(pickledFilePath, str):
        with open(pickledFilePath) as f:
            ob = pickle.load(f)
            f.close()
    else:
        ob = pickledFilePath
    train_params, train_y, test_params, test_y = ob
    allParams = np.concatenate((train_params, test_params), axis=0)
    
    if external_lables is None:
        allLabels = np.concatenate((train_y, test_y), axis=0)
    else:
        allLabels = external_lables[:train_params.shape[0] + test_params.shape[0]]
    
    return allLabels, allParams

def checkLabelPredict(allLabels,allParams,labelNumber=0,NUM_SPLITS = 5):

    print "Label- ", labelNumber
    
    negative_labels=np.zeros(allLabels.shape[0])
    positive_labels=np.zeros(allLabels.shape[0])
    negative_params=[] #
#     negative_params=np.zeros(shape=(allParams.shape[0],allParams.shape[1]))
    positive_params=[] #
#     positive_params=np.zeros(shape=(allParams.shape[0],allParams.shape[1]))
    countNegatives = 0
    countPositives = 0
    for example in range(0,allParams.shape[0]):
        curLabel = allLabels[example][labelNumber]
        if curLabel==0:
            negative_labels[countNegatives] = curLabel
#             negative_params[countNegatives] = allParams[example]
            negative_params.append(allParams[example])
            countNegatives += 1
        else:
            positive_labels[countPositives] = curLabel
#             positive_labels[countPositives] = allParams[example]
            positive_params.append(allParams[example])
            countPositives += 1
    negative_labels = negative_labels[:countNegatives]
    positive_labels = positive_labels[:countPositives]
#     negative_params = negative_params[:countNegatives]
#     positive_params = positive_params[:countPositives]
    negative_params = np.array(negative_params)
    positive_params = np.array(positive_params)
#     
    print "    Positives-", countPositives
    print "    Negatives-", countNegatives
    
    if countPositives <=0 :
        return 100 #bad error because svm can't run on one catagory
    
    errorRates = np.zeros(NUM_SPLITS)
    aucScores = np.zeros(NUM_SPLITS)
    negativeDataPart = np.floor(negative_labels.shape[0]/NUM_SPLITS)
    positiveDataPart = np.floor(positive_labels.shape[0]/NUM_SPLITS)
    for i in range(0,NUM_SPLITS):
        if i==0:
            train_labels = np.concatenate((negative_labels[negativeDataPart:],positive_labels[positiveDataPart:]),axis=0)
            test_labels = np.concatenate((negative_labels[:negativeDataPart],positive_labels[:positiveDataPart]),axis=0)
            train_params = np.concatenate((negative_params[negativeDataPart:],positive_params[positiveDataPart:]),axis=0)
            test_params = np.concatenate((negative_params[:negativeDataPart],positive_params[:positiveDataPart]),axis=0)
        elif i==NUM_SPLITS-1:
            if positiveDataPart>0:
                train_labels = np.concatenate((negative_labels[:(NUM_SPLITS-1)*negativeDataPart],positive_labels[:(NUM_SPLITS-1)*positiveDataPart]),axis=0)
                test_labels = np.concatenate((negative_labels[(NUM_SPLITS-1)*negativeDataPart:],positive_labels[(NUM_SPLITS-1)*positiveDataPart:]),axis=0)
                train_params = np.concatenate((negative_params[:(NUM_SPLITS-1)*negativeDataPart],positive_params[:(NUM_SPLITS-1)*positiveDataPart]),axis=0)
                test_params = np.concatenate((negative_params[(NUM_SPLITS-1)*negativeDataPart:],positive_params[(NUM_SPLITS-1)*positiveDataPart:]),axis=0)
            else:
                #train set has to have a positive example
                train_labels = np.concatenate((negative_labels[:(NUM_SPLITS-1)*negativeDataPart],positive_labels),axis=0)
                test_labels = negative_labels[(NUM_SPLITS-1)*negativeDataPart:]
                train_params = np.concatenate((negative_params[:(NUM_SPLITS-1)*negativeDataPart],positive_params),axis=0)
                test_params = negative_params[(NUM_SPLITS-1)*negativeDataPart:]
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
        auc_score = roc_auc_score(test_labels, test_predict)

        
        print "        Split number- ", i
        print differ 
        print "        Error- ", error_rate, "%"
        
        print "        RocAucScore- ", auc_score       
        
        errorRates[i] = error_rate
        aucScores[i] = auc_score
    
    return (np.average(errorRates),np.average(aucScores))

   
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
          
def runCrossSvm(pickName,num_labels = 20):
    
#     labelDir = "C:\\Users\\Ido\\workspace\\ISH_Lasagne\\articleCatagorise\\*articleCatagorise.txt"
#     labelDir = "C:\\Users\\Ido\\Pictures\\BrainISHimages\\*TopCat.txt"
#     y = readNewLables(labelDir,end_index=16351)

    if num_labels==15:
        y = loadNewLabels("C:\\Users\\Ido\\workspace\\ISH_Lasagne\\src\\DeepLearning\\pickled_images\\articleCat.pkl.gz")
    elif num_labels==20:
        y = loadNewLabels("C:\\Users\\Ido\\workspace\\ISH_Lasagne\\src\\DeepLearning\\pickled_images\\topCat.pkl.gz")
    elif num_labels==164:
        y = loadNewLabels("C:\\Users\\Ido\\workspace\\ISH_Lasagne\\src\\DeepLearning\\pickled_images\\all164cat.pkl.gz")
    else:
        y = loadNewLabels("C:\\Users\\Ido\\workspace\\ISH_Lasagne\\src\\DeepLearning\\pickled_images\\all2081cat.pkl.gz")
         
    allLabels, allParams = getData(pickName,y)

    errorRates = np.zeros(num_labels)
    aucScores = np.zeros(num_labels)
    for i in range(0,num_labels):
        labelPredRate,labelAucScore = checkLabelPredict(allLabels, allParams,labelNumber=i) 
        errorRates[i] = labelPredRate
        aucScores[i] = labelAucScore
    
    errorRate = np.average(errorRates)
    aucAverageScore = np.average(aucScores)
    print "Average error- ", errorRate, "%"
    print "Prediction rate- ", 100-errorRate, "%"
    print "Average Auc Score- ", aucAverageScore
    
    return (errorRates,aucScores)
    
if __name__ == "__main__":
#     pickName = "C:\\Users\\Ido\\workspace\\ISH_Lasagne\\src\\DeepLearning\\results\\noLearn_50_3_hiddenLayerOutput_0.pickle"
#     pickName = "C:\\Users\\Ido\\workspace\\ISH_Lasagne\\src\\DeepLearning\\results\\ISH-noLearn_0_5000_300_140\\run_0\\hiddenLayerOutput.pickle"
    pickName = "C:\\Users\\Ido\\workspace\\ISH_Lasagne\\src\\DeepLearning\\results\\articleCat\\11000_pic\\run_1\\hiddenLayerOutput.pickle"
    runCrossSvm(pickName,15)

  
