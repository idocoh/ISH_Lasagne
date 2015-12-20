# http://scikit-learn.org/stable/tutorial/basic/tutorial.html
from sklearn import svm
from sklearn import datasets
import cPickle as pickle
import numpy as np

def checkLabelPredict(pcikledFilePath, outputFileName,labelNumber=0):
    with open(pcikledFilePath) as f:
        ob = pickle.load(f)
        f.close()
    train_params, train_labels, test_params, test_labels = ob;
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
            
          

    
if __name__ == "__main__":
    pickName = "C:\\Users\\Ido\\workspace\\ISH_Lasagne\\src\\DeepLearning\\results\\noLearn_50_3_hiddenLayerOutput_0.pickle"
    num_labels = 20
    sum_error = 0
    for i in range(0,num_labels):
        sum_error+=checkLabelPredict(pickName,"myData.txt",i)
    
    print "Average error- ", sum_error/num_labels, "%"
    print "Prediction rate- ", 100-sum_error/num_labels, "%"

  