# http://scikit-learn.org/stable/tutorial/basic/tutorial.html
from sklearn import svm
from sklearn import datasets
import cPickle as pickle
import numpy as np
from pickleImages import getTopCatVector
from numpy.f2py.auxfuncs import isstring
from sklearn.metrics import roc_auc_score
import sklearn.svm.libsvm as svm
import gzip

def checkLabelPredict(pickledFilePath,labelNumber=0,external_lables=None):
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
    
    if external_lables is not None:
        train_labels = external_lables[:train_params.shape[0]]
        test_labels = external_lables[train_params.shape[0]:train_params.shape[0]+test_params.shape[0]]
    
    train_y=np.zeros(train_params.shape[0])
    for lb in range(0,train_labels.shape[0]):
        train_y[lb]=train_labels[lb][labelNumber]
    
    test_y=np.zeros(test_params.shape[0])
    for lb in range(0,test_params.shape[0]):
        test_y[lb]=test_labels[lb][labelNumber]
    
    error_rate=-100
    auc_score=-100
    
    try:
#                 
#         function [] = run_svms_rebuttal(all_labels, bow, filename, i_cat, ...
#                                         inner_train_inds, inner_validation_inds, k, C_vals)
#         % run SVM on samples
#         %
#         % d is the dimension, n is the number of instances
#         % X is a dxn sparse matrix of features
#         % bin_labels is an nx1 full binary label vector
#         % C_vals is the magnitude of the C parameter in the SVM
#         
#         % k=5;  %%%GGG<==== Pass this as a parameter
#         % C_vals = 10.^[-6:3];
#         % C_vals = 10.^[-3:2]; %%%GGG<==== Pass this as a parameter
#         
#         
#         p_validation = cell(k, k, length(C_vals));
#         scores_validation = cell(k, k, length(C_vals));
#         auc = nan(k, k, length(C_vals));
#         
#         t = cputime;
#         for i_outer = 1:k
#             % fprintf('i_outer = %d\n', i_outer);
#         
#             for i_inner = 1:k
#                 fprintf('i_outer = %d, i_inner = %d, cputime =  %3.2g min [%s]\n', i_outer, i_inner, ...
#                     (cputime-t)/60, datestr(now))
#         
#                 X_train = bow(inner_train_inds{i_outer, i_inner}, :);
#                 X_validation = bow(inner_validation_inds{i_outer, i_inner}, :);
#         
#                 train_bin_labels = all_labels(inner_train_inds{i_outer, i_inner});
#                 validation_bin_labels = all_labels(inner_validation_inds{i_outer, i_inner});
#                 fprintf('train positives = %d, validation positives = %d\n', ...
#                         sum(train_bin_labels), sum(validation_bin_labels))
#                 if sum(validation_bin_labels)~=0
#                     neg_pos_ratio = (size(train_bin_labels,1)-sum(train_bin_labels))/sum(train_bin_labels);
#                     if ~isinf(neg_pos_ratio)
#                         for i_param = 1:length(C_vals)
#                             % get model + train and validation prediction. save model
#                             % and predictions
#         
#                             svm_opt = sprintf('-w1 %g -w0 1 -c %g  -s 0 -B 1 -q 1 ',neg_pos_ratio, C_vals(i_param));                                                                                                                               
#
#                             fprintf('before train %3.2g min [%s]\n', ...
#                                     (cputime-t)/60, datestr(now))
#                             model =train([],train_bin_labels,X_train,svm_opt);
#         
#                             fprintf('before predict %3.2g min [%s]\n', ...
#                                     (cputime-t)/60, datestr(now))
#                             [p_validation{i_outer, i_inner, i_param},~, temp_scores] =...
#                                 predict(validation_bin_labels,X_validation,model,'-b 1');
#                             scores_validation{i_outer, i_inner, i_param} = temp_scores(:,end-model.Label(1));
#                             auc(i_outer, i_inner, i_param) = ...
#                                 auc_uri(logical(validation_bin_labels), ...
#                                         scores_validation{i_outer, i_inner, i_param}, 0);
#                             fprintf('auc = %.3g, C val = %.3g\n',auc(i_outer, i_inner, i_param),...
#                                     C_vals(i_param));
#                             fprintf('after predict %3.2g min [%s]\n', ...
#                                     (cputime-t)/60, datestr(now))
#                         end
#         
#                     end
#                 end
#             end
#         end
#         
#         these_scores = scores_validation;
#         these_p = p_validation;
#         these_auc = auc;
#         svm_parsave(sprintf('%s_%d.mat',filename,i_cat), these_p, these_scores, these_auc);
#         fprintf('saving to filename %s\n',sprintf('%s_%d.mat',filename,i_cat));
#         e = cputime-t;
        
        neg_pos_ratio = (len(train_y)-np.sum(train_y))/np.sum(train_y)
        
        C_vals = [10^-3,10^-2,10^-1,10^0,10^1,10^2]
        for i_param in range (0,len(C_vals)):
            prob  = svm_problem(train_y, train_params, isKernel=True)
            param = svm_parameter('-w1 '+neg_pos_ratio+' -w0 1 -c '+C_vals(i_param)+'  -s 0 -B 1 -q 1 ')
            m = svm_train(prob, param)                 
    #         svm_save_model('heart_scale.model', m)
    #         m = svm_load_model('heart_scale.model')
            test_predict, p_acc, p_val = svm_predict(test_y, test_params, m, '-b 1')
            ACC, MSE, SCC = evaluations(train_y, test_predict)
#         clf = svm.SVC()
#         clf.fit(train_params, train_y)
#         
#         test_predict = clf.predict(test_params);
        
        differ = test_predict-test_y
        error_rate = np.sum(np.abs(differ))/differ.shape[0] * 100
        auc_score = roc_auc_score(test_y, test_predict)
    
        print differ 
        print "        Error- ", error_rate, "%"
        print "            RocAucScore- ", auc_score       
    
    except:
        print "                    bad label"
    return (error_rate, auc_score)
   
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
          
def runSvm(pickName,num_labels=20, readLabelsAgain = False):
    
#     labelDir = "C:\\Users\\Ido\\workspace\\ISH_Lasagne\\articleCatagorise\\*articleCatagorise.txt"
#     labelDir = "C:\\Users\\Ido\\Pictures\\BrainISHimages\\*TopCat.txt"
#     y = readNewLables(labelDir,end_index=16351)

    if not readLabelsAgain:
        y = None
    elif num_labels==15:
        y = loadNewLabels("C:\\Users\\Ido\\workspace\\ISH_Lasagne\\src\\DeepLearning\\pickled_images\\articleCat.pkl.gz")
    elif num_labels==20:
        y = loadNewLabels("C:\\Users\\Ido\\workspace\\ISH_Lasagne\\src\\DeepLearning\\pickled_images\\topCat.pkl.gz")
    
    errorRates = np.zeros(num_labels)
    aucScores = np.zeros(num_labels)
    for i in range(0,num_labels):
        labelPredRate,labelAucScore = checkLabelPredict(pickName,i,y) 
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
    pickName = "C:\\Users\\Owner\\workspace\\ISH_Lasagne\\src\\DeepLearning\\results_dae\\try_0.2\\run_0\\hiddenLayerOutput.pkl.gz"
    runSvm(pickName,15)
     