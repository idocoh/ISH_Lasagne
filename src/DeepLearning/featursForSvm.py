from __future__ import print_function
import os
import cPickle as pickle
import gzip
import numpy as np
from sklearn import svm
from sklearn.metrics import roc_auc_score
from pickleForArticleCat import pickleAllImages
from nolearn.lasagne import BatchIterator
from lasagne import layers
from theano import config

from nolearn.lasagne import NeuralNet
from shape import ReshapeLayer
from lasagne.nonlinearities import tanh
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import TrainSplit
import time

# import liblinear
from liblinearutil import *

COMPACT_VECTOR_SIZE = 800

CONV_AE_PARAMS_PKL = 'conv_ae_params.pkl'
CONV_AE_NP = 'conv_ae.np'
CONV_AE_PKL = 'conv_ae.pkl'

class Unpool2DLayer(layers.Layer):
    """
    This layer performs unpooling over the last two dimensions
    of a 4D tensor.
    """
    def __init__(self, incoming, ds, **kwargs):

        super(Unpool2DLayer, self).__init__(incoming, **kwargs)

        if (isinstance(ds, int)):
            raise ValueError('ds must have len == 2')
        else:
            ds = tuple(ds)
            if len(ds) != 2:
                raise ValueError('ds must have len == 2')
            if ds[0] != ds[1]:
                raise ValueError('ds should be symmetric (I am lazy)')
            self.ds = ds

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)

        output_shape[2] = input_shape[2] * self.ds[0]
        output_shape[3] = input_shape[3] * self.ds[1]

        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        ds = self.ds
        input_shape = input.shape
        output_shape = self.get_output_shape_for(input_shape)
        return input.repeat(2, axis=2).repeat(2, axis=3)


### when we load the batches to input to the neural network, we randomly / flip
### rotate the images, to artificially increase the size of the training set
class FlipBatchIterator(BatchIterator):
    def transform(self, X1, X2):
        X1b, X2b = super(FlipBatchIterator, self).transform(X1, X2)
        X2b = X2b.reshape(X1b.shape)

        bs = X1b.shape[0]
        h_indices = np.random.choice(bs, bs / 2, replace=False)  # horizontal flip
        v_indices = np.random.choice(bs, bs / 2, replace=False)  # vertical flip

        ###  uncomment these lines if you want to include rotations (images must be square)  ###
        #r_indices = np.random.choice(bs, bs / 2, replace=False) # 90 degree rotation
        for X in (X1b, X2b):
            X[h_indices] = X[h_indices, :, :, ::-1]
            X[v_indices] = X[v_indices, :, ::-1, :]
            #X[r_indices] = np.swapaxes(X[r_indices, :, :, :], 2, 3)
        shape = X2b.shape
        X2b = X2b.reshape((shape[0], -1))

        return X1b, X2b


def images_svm(pickled_file, x=None, all_labels=None, svm_negative_amount=800, num_labels=15, TRAIN_SPLIT=0.8):

    if isinstance(pickled_file, str):
        try:
            features = pickle.load(open(pickled_file, 'r'))
        except:
            f = gzip.open(pickled_file, 'rb')
            features = pickle.load(f)
            f.close()
    else:
        cnn = pickled_file
        if x is None:
            print ("Reading images...")
            all_labels, x = pickleAllImages(num_labels=15, pos=True)
        print("Processing images...")
        # input_width, input_height, dropout_percent = 300, 140, 0.2
        # x = x.astype(np.float32).reshape((-1, 1, input_width, input_height))
        # x *= np.random.binomial(1, 1 - dropout_percent, size=x.shape)
        # features = cnn.output_hiddenLayer(x)

        x, all_labels = separate_svm(x.astype(np.float32), all_labels, svm_negative_amount)
        x = x
        all_labels = all_labels

        start_time = time.clock()
        print("Starting cnn prediction...")
        # (w, l) = cnn.output_hiddenLayer(x[1:2]).reshape((1, -1)).shape
        # COMPACT_VECTOR_SIZE = w*l
        features = np.zeros((x.shape[0], COMPACT_VECTOR_SIZE))
        # features = []
        for i in range(0, x.shape[0]):
            features[i:i+1, :] = cnn.output_hiddenLayer(x[i:i+1]).reshape((1, -1))
            # features[i] = cnn.output_hiddenLayer(x[i:i+1]).reshape((1, -1))
        # quarter_x = np.floor(x.shape[0] / 4)
        # train_last_hidden_layer_1 = cnn.output_hiddenLayer(x[:quarter_x])
        # train_last_hidden_layer_1 = train_last_hidden_layer_1.reshape((train_last_hidden_layer_1.shape[0], -1))
        # print("after first quarter train output")
        # train_last_hidden_layer_2 = cnn.output_hiddenLayer(x[quarter_x:2 * quarter_x])
        # train_last_hidden_layer_2 = train_last_hidden_layer_2.reshape((train_last_hidden_layer_2.shape[0], -1))
        # print("after second quarter train output")
        # train_last_hidden_layer_3 = cnn.output_hiddenLayer(x[2 * quarter_x: 3 * quarter_x])
        # train_last_hidden_layer_3 = train_last_hidden_layer_3.reshape((train_last_hidden_layer_3.shape[0], -1))
        # print("after third quarter train output")
        # train_last_hidden_layer_4 = cnn.output_hiddenLayer(x[3 * quarter_x:])
        # train_last_hidden_layer_4 = train_last_hidden_layer_4.reshape((train_last_hidden_layer_4.shape[0], -1))
        # features = np.concatenate((train_last_hidden_layer_1, train_last_hidden_layer_2, train_last_hidden_layer_3, train_last_hidden_layer_4), axis=0)

        # print('Features size: ', features.shape)
        # print('Labels size: ', all_labels.shape)
    run_time = (time.clock() - start_time) / 60.
    print("     CNN prediction took(min)- ", run_time)

    # if num_labels==15:
    #     labels_file = "pickled_images"+FILE_SEPARATOR+"articleCat.pkl.gz"
    # elif num_labels==20:
    #     labels_file = "pickled_images"+FILE_SEPARATOR+"topCat.pkl.gz"
    # elif num_labels==164:
    #     labels_file = "pickled_images"+FILE_SEPARATOR+"all164cat.pkl.gz"
    # elif num_labels==2081:
    #     labels_file = "pickled_images"+FILE_SEPARATOR+"all2081cat.pkl.gz"
    # else:
    #     print("bad labels path!!!!!!!")
    #
    # print("Loading labels")
    # with open(labels_file) as l:
    #     labels = pickle.load(l)[:features.shape[0], :]
    #     l.close()

    return features, all_labels


def separate_svm(pData, pLabel, svm_negative_amount):
    posRows = (pLabel != 0).sum(1) > 0
    posData = pData[posRows, :]
    posLabel = pLabel[posRows, :]
    print("Positive svm samples- ", posData.shape[0])
    negData = pData[~posRows[:svm_negative_amount], :]
    negLabel = pLabel[~posRows[:svm_negative_amount], :]
    print("Negative svm samples- ", negData.shape[0])
    # svm_data = np.concatenate((posData, negData[:svm_size-posData.shape[0]]), axis=0)
    # svm_label = np.concatenate((posLabel, negLabel[:svm_size-posData.shape[0]]), axis=0)
    svm_data = np.concatenate((posData, negData), axis=0)
    svm_label = np.concatenate((posLabel, negLabel), axis=0)
    return svm_data, svm_label


def recunstruct_cae(folder_path):
    cnn = NeuralNet()
    cnn.load_params_from(folder_path + CONV_AE_PARAMS_PKL)
    cnn.load_weights_from(folder_path + CONV_AE_NP)
    return cnn


def classifier_score(neg_test, neg_train, pos_test, pos_train):
    # NN_classifier_score(neg_test, neg_train, pos_test, pos_train)
    # svc_score(neg_test, neg_train, pos_test, pos_train)
    # return svc_score(neg_test, neg_train, pos_test, pos_train)
    # return linear_svc_score(neg_test, neg_train, pos_test, pos_train)
    return lib_linear_score(neg_test, neg_train, pos_test, pos_train)


def checkLabelPredict(features, labels, cross_validation_parts=5):
    try:
        print ("Features size- ", features.shape)
        print ("Size of positive samples- ", (features[labels == 1]).shape)
    except:
        pass
    positive_data = features[labels == 1, :]
    # positive_data = positive_data.reshape((-1, features.shape[-2]*features.shape[-1]))
    negative_data = features[labels == 0, :]
    # negative_data = negative_data.reshape((-1, features.shape[-2]*features.shape[-1]))

    if positive_data.shape[0] < cross_validation_parts or negative_data.shape[0] < cross_validation_parts:
        return -1, -1

    negative_data_chunks = np.array_split(negative_data, cross_validation_parts)
    positive_data_chunks = np.array_split(positive_data, cross_validation_parts)

    scores = np.zeros(cross_validation_parts)
    auc_scores = np.zeros(cross_validation_parts)

    for cross_validation_index in range(0, cross_validation_parts):
        neg_test = negative_data_chunks[cross_validation_index]
        neg_train = np.copy(negative_data_chunks)
        np.delete(neg_train, cross_validation_index)
        neg_train = np.concatenate(neg_train)
        print("     Number of negative train- ", neg_train.shape[0], " test- ", neg_test.shape[0])

        pos_test = positive_data_chunks[cross_validation_index]
        pos_test = generate_positives(pos_test, neg_test.shape[0])

        pos_train = np.copy(positive_data_chunks)
        np.delete(pos_train, cross_validation_index)
        pos_train = np.concatenate(pos_train)
        print("     Number of positive train- ", pos_train.shape[0])
        pos_train = generate_positives(pos_train, neg_train.shape[0])
        print("         Number of generated positive train- ", pos_train.shape[0], " test- ", pos_test.shape[0])

        # clf, score, test_params, test_y = classifier_score(neg_test, neg_train, pos_test, pos_train)

        clf, score, test_params, test_y = lib_linear_score(neg_test, neg_train, pos_test, pos_train)
        scores[cross_validation_index] = score
        clf_svm, score_svm, test_params_svm, test_y_svm = svc_score(neg_test, neg_train, pos_test, pos_train)
        auc_scores[cross_validation_index] = score_svm

        # try:
        #     test_predict = clf.predict(test_params)
        #     auc_score = roc_auc_score(test_y, test_predict)
        #     print("AUC cross-", auc_score)
        #     auc_scores[cross_validation_index] = auc_score
        # except Exception as e:
        #     print(e)
        #     print(e.message)

    try:
        return np.average(scores), np.average(auc_scores)
    except:
        return np.average(scores), 999



def svc_score(neg_test, neg_train, pos_test, pos_train):
    clf = svm.SVC(kernel='linear', C=1).fit(np.concatenate((pos_train, neg_train), axis=0),
                                            np.concatenate((np.ones(pos_train.shape[0]),
                                                            np.zeros(neg_train.shape[0])), axis=0))
    test_params = np.concatenate((pos_test, neg_test), axis=0)
    test_y = np.concatenate((np.ones(pos_test.shape[0]), np.zeros(neg_test.shape[0])), axis=0)
    score = clf.score(test_params, test_y)
    print("SVM score- ", score)
    return clf, score, test_params, test_y


def linear_svc_score(neg_test, neg_train, pos_test, pos_train):
    clf = svm.LinearSVC(C=1).fit(np.concatenate((pos_train, neg_train), axis=0),
                                            np.concatenate((np.ones(pos_train.shape[0]),
                                                            np.zeros(neg_train.shape[0])), axis=0))
    test_params = np.concatenate((pos_test, neg_test), axis=0)
    test_y = np.concatenate((np.ones(pos_test.shape[0]), np.zeros(neg_test.shape[0])), axis=0)
    score = clf.score(test_params, test_y)
    print("SVM score- ", score)
    return clf, score, test_params, test_y

def lib_linear_score(neg_test, neg_train, pos_test, pos_train):
    y = np.concatenate((np.ones(pos_train.shape[0]), -1*np.ones(neg_train.shape[0])), axis=0)
    x = np.concatenate((pos_train, neg_train), axis=0)
    clf = train(y.tolist(), x.tolist(), '-c 1 -s 0')
    test_params = np.concatenate((pos_test, neg_test), axis=0)
    test_y = np.concatenate((np.ones(pos_test.shape[0]), -1*np.ones(neg_test.shape[0])), axis=0)
    p_label, p_acc, p_val = predict(test_y.tolist(), test_params.tolist(), clf)
    '''
    p_acc: a tuple including  accuracy (for classification), mean-squared
    error, and squared correlation coefficient (for regression).
    '''
    print("mean-squared error- ", p_acc[1])
    print("squared correlation coefficient- ", p_acc[2])

    score = roc_auc_score(test_y, np.array(p_label))
    print("AUC- ", score)

    return clf, score, test_params, test_y

    # prob = liblinearutil.problem([1, -1], [{1: 1, 3: 1}, {1: -1, 3: -1}])
    # param = parameter('-c 4')
    # m = liblinear.train(prob, param)
    # x0, max_idx = gen_feature_nodearray({1: 1, 3: 1})
    # label = liblinear.predict(m, x0)

    # clf = svm.LinearSVC(C=1).fit(np.concatenate((pos_train, neg_train), axis=0),
    #                                         np.concatenate((np.ones(pos_train.shape[0]),
    #                                                         np.zeros(neg_train.shape[0])), axis=0))
    # test_params = np.concatenate((pos_test, neg_test), axis=0)
    # test_y = np.concatenate((np.ones(pos_test.shape[0]), np.zeros(neg_test.shape[0])), axis=0)
    # score = clf.score(test_params, test_y)
    # print("SVM score- ", score)
    # return clf, score, test_params, test_y

def NN_classifier_score(neg_test, neg_train, pos_test, pos_train):
    y = np.concatenate((np.ones(pos_train.shape[0]), -1*np.ones(neg_train.shape[0])), axis=0)
    x = np.concatenate((pos_train, neg_train), axis=0)
    # clf = train(y.tolist(), x.tolist(), '-c 1 -s 0')

    test_params = np.concatenate((pos_test, neg_test), axis=0)
    test_y = np.concatenate((np.ones(pos_test.shape[0]), -1*np.ones(neg_test.shape[0])), axis=0)
    # p_label, p_acc, p_val = predict(test_y.tolist(), test_params.tolist(), clf)

    import nn_classifier
    nn_classifier.main(x, y, test_y, test_params)

    # score = roc_auc_score(test_y, np.array(p_label))
    # print("NN AUC- ", score)
    #
    # return clf, score, test_params, test_y

def generate_positives(positives, num_negatives):
    num_positives = positives.shape[0]
    multiple_by = np.ones(num_positives)*np.divide(num_negatives, num_positives)
    for i in range(0, np.remainder(num_negatives, num_positives)):
        multiple_by[i] += 1

    return np.repeat(positives, multiple_by.astype(int), axis=0)


def run_svm(pickle_name, X_train=None, labels=None, svm_negative_amount=800):
    num_labels = 15
    features, labels = images_svm(pickle_name, X_train, labels,  num_labels=num_labels, svm_negative_amount=svm_negative_amount)

    errorRates = np.zeros(num_labels)
    aucScores = np.zeros(num_labels)

    start_time = time.clock()
    for label in range(0, num_labels):
        print("Svm for category- ", label)
        labelPredRate, labelAucScore = checkLabelPredict(features, labels[:, label])
        errorRates[label] = labelPredRate
        aucScores[label] = labelAucScore
        print("Average category error- ", labelPredRate, "%")
        print("Average category Auc Score- ", labelAucScore)

    errorRate = np.average(errorRates)
    aucAverageScore = np.average(aucScores)
    print("Average error- ", errorRate, "%")
    # print("Prediction rate- ", 100 - errorRate, "%")
    print("Average Auc Score- ", aucAverageScore)
    run_time = (time.clock() - start_time) / 60.
    print("SVM took(min)- ", run_time)

    # try:
    #     # pickle.dump((features, labels), open('svm.pkl', 'w'))
    #     print("Trying to pickle svm... ")
    #     pickle.dump(X_train, open('svm-x-data.pkl', 'w'))
    #     pickle.dump(labels, open('svm--data.pkl', 'w'))
    # except:
    #     pass

    return errorRates, aucScores

if __name__ == '__main__':
    # pickName = "C:/devl/python/ISH_Lasagne/src/DeepLearning/results_dae/learn/run_0/encode.pkl"
    pickled_file = "C:/devl/python/ISH_Lasagne/src/DeepLearning/results_dae/for_debug/run_0/conv_ae.pkl"
    net = pickle.load(open(pickled_file, 'r'))
    run_svm(net)




