# http://scikit-learn.org/stable/tutorial/basic/tutorial.html
from sklearn import svm
from sklearn import datasets
import cPickle as pickle
import numpy as np
from pickleImages import getTopCatVector
from numpy.f2py.auxfuncs import isstring

# articleCatNames = {
#     "Negative regulation of elastin catabolic process",
#     "Long-chain fatty acid biosynthetic process",
#     "-Aminobutyric acid biosynthetic process",
#     "-Aminobutyric acid metabolic process",
#     "Negative reg. of aldosterone biosynthetic process",
#     "Negative regulation o""f cortisol biosynthetic process",
#     "Fibril organization",
#     "Negative reg. of glucocorticoid biosynthetic process",
#     "Neurotransmitter biosynthetic process",
#     "Central nervous system myelination",
#     "Neuron recognition",
#     "Response to cocaine",
#     "Negative chemotaxis",
#     "Ribosomal small subunit biogenesis",
#     "Peptide hormone processing",
#     }

def checkLabelPredict(pickledFilePath,labelNumber=0,external_lables=None):
    if isstring(pickledFilePath):
        with open(pickledFilePath) as f:
            ob = pickle.load(f)
            f.close()
    else:
        ob=pickledFilePath
    train_params, train_labels, test_params, test_labels = ob;
    
    if external_lables is not None:
        train_labels = external_lables[:train_params.shape[0]]
        test_labels = external_lables[train_params.shape[0]:train_params.shape[0]+test_params.shape[0]]
    
    y=np.zeros(train_params.shape[0])
    for lb in range(0,train_labels.shape[0]):
        y[lb]=train_labels[lb][labelNumber]
    
    test_y=np.zeros(test_params.shape[0])
    for lb in range(0,test_params.shape[0]):
        test_y[lb]=test_labels[lb][labelNumber]
    
    clf = svm.SVC()
    clf.fit(train_params, y)
    
    test_predict = clf.predict(test_params);
    
    differ = test_predict-test_y
    error_rate = np.sum(np.abs(differ))/differ.shape[0] * 100
    
    print "Predict label- ", labelNumber
    print differ 
    print "Error- ", error_rate, "%"
    return error_rate
   
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

    if num_labels==15:
        y = loadNewLabels("C:\\Users\\Ido\\workspace\\ISH_Lasagne\\src\\DeepLearning\\pickled_images\\articleCat.pkl.gz")
    else:
        y = loadNewLabels("C:\\Users\\Ido\\workspace\\ISH_Lasagne\\src\\DeepLearning\\pickled_images\\topCat.pkl.gz")
    
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
     