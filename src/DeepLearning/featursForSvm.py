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
import nnClassifier

pickled_folder = "C:/devl/work/ISH_Lasagne/src/DeepLearning/results_dae/CAE_16351_different_sizes-1489653160.29/run_4"
results_file = open(pickled_folder + "/NN_output.txt", "a")

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
            print("Reading images...")
            all_labels, x = pickleAllImages(num_labels=15, pos=True)
        print("Processing images...")
        # input_width, input_height, dropout_percent = 300, 140, 0.2
        # x = x.astype(np.float32).reshape((-1, 1, input_width, input_height))
        # x *= np.random.binomial(1, 1 - dropout_percent, size=x.shape)
        # features = cnn.output_hiddenLayer(x)

        x, all_labels = separate_svm(x.astype(np.float32), all_labels, svm_negative_amount)

        start_time = time.clock()
        print("Starting cnn prediction...")
        (w, l) = cnn.output_hiddenLayer(x[1:2]).reshape((1, -1)).shape
        features = np.zeros((x.shape[0], w * l)).astype(np.float32)
        # features = []
        for i in range(0, x.shape[0]):
            features[i:i + 1, :] = cnn.output_hiddenLayer(x[i:i + 1]).reshape((1, -1)).astype(np.float32)

    run_time = (time.clock() - start_time) / 60.
    print("     CNN prediction took(min)- ", run_time)

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
    CONV_AE_PARAMS_PKL = 'conv_ae_params.pkl'
    CONV_AE_NP = 'conv_ae.np'
    CONV_AE_PKL = 'conv_ae.pkl'
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
        print("Features size- ", features.shape)
        print("Size of positive samples- ", (features[labels == 1]).shape)
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
        results_file.write("Cross validation #" + str(cross_validation_index+1) + "\n")
        results_file.flush()

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

        # Test: change
        clf, score, test_params, test_y = NN_classifier_score(neg_test, neg_train, pos_test, pos_train)
        scores[cross_validation_index] = score

        # clf, score, test_params, test_y = lib_linear_score(neg_test, neg_train, pos_test, pos_train)
        # scores[cross_validation_index] = score
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
    y = np.concatenate((np.ones(pos_train.shape[0]), -1 * np.ones(neg_train.shape[0])), axis=0)
    x = np.concatenate((pos_train, neg_train), axis=0)
    clf = train(y.tolist(), x.tolist(), '-c 1 -s 0')
    test_params = np.concatenate((pos_test, neg_test), axis=0)
    test_y = np.concatenate((np.ones(pos_test.shape[0]), -1 * np.ones(neg_test.shape[0])), axis=0)
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


def NN_classifier_score(neg_test, neg_train, pos_test, pos_train):
    pos_train_size = pos_train.shape[0]
    neg_train_size = neg_train.shape[0]
    y = np.transpose(np.concatenate(
        ([np.concatenate((np.ones(pos_train_size, np.float32), 0 * np.ones(neg_train_size, np.float32)), axis=0)],
         [np.concatenate((0 * np.ones(pos_train_size, np.float32), np.ones(neg_train_size, np.float32)), axis=0)]),
        axis=0))
    pos_test_size = pos_test.shape[0]
    neg_test_size = neg_test.shape[0]
    test_y = np.transpose(np.concatenate(
        ([np.concatenate((np.ones(pos_test_size, np.float32), 0 * np.ones(neg_test_size, np.float32)), axis=0)],
         [np.concatenate((0 * np.ones(pos_test_size, np.float32), np.ones(neg_test_size, np.float32)), axis=0)]),
        axis=0))
    x = np.concatenate((pos_train, neg_train), axis=0)
    test_params = np.concatenate((pos_test, neg_test), axis=0)
    print ("done with data to nn, training...")
    return try_nn(test_params, test_y, x, y)


def try_nn(test_params, test_y, x, y):
    layers_size = [
        # [1000, 100],
        # [750, 250],
        # [500, 100],
        [1000, 300],
        # [1000, 250, 50],
        [1000, 500, 250],
        [2000, 1000, 500],
        # [1200, 800, 400],
        [2000, 1000, 500, 250],
        [2000, 1000, 500, 250, 50]
    ]

    temp_auc = np.zeros((3, 5))
    for j in range(0, 5):
        try:
            for i in range(0, 3):
                try:
                    learning_rate = 0.01 + 0.02 * i
                    classifier_net, error_rate, auc_score = \
                        nnClassifier.runNNclassifier(x, y, test_params, test_y, LEARNING_RATE=learning_rate,
                                                     NUM_UNITS_HIDDEN_LAYER=layers_size[j])
                    temp_auc[i, j] = auc_score
                    print("AUC- " + str(auc_score) + ": rate " + str(learning_rate) + ", layers " + str(layers_size[j]))
                    results_file.write("AUC- " + str(auc_score) + ", rate- " + str(learning_rate) + ", layers- " + str(
                        layers_size[j]) + "\n")
                    results_file.flush()
                except Exception as e:
                    print("failed nn i-", i)
                    print(e)
                    print(e.message)
            results_file.write("Max AUC- " + str(np.max(temp_auc[:, j])) + ", rate- all, layers- " + str(
                layers_size[j]) + "\n")
            results_file.flush()
        except Exception as e:
            print("failed nn i-", i)
            print(e)
            print(e.message)

    return classifier_net, np.max(temp_auc[i, j]), test_params, test_y


def generate_positives(positives, num_negatives):
    num_positives = positives.shape[0]
    multiple_by = np.ones(num_positives) * np.divide(num_negatives, num_positives)
    for i in range(0, np.remainder(num_negatives, num_positives)):
        multiple_by[i] += 1

    return np.repeat(positives, multiple_by.astype(int), axis=0)


def run_svm(pickle_name=None, X_train=None, features=None, labels=None, svm_negative_amount=800, folder_path=None):
    num_labels = labels.shape[1]
    if features is None:
        features, labels = images_svm(pickle_name, X_train, labels, num_labels=num_labels,
                                      svm_negative_amount=svm_negative_amount)

    errorRates = np.zeros(num_labels)
    aucScores = np.zeros(num_labels)

    start_time = time.clock()
    # Test"
    for label in range(0, num_labels):
    # for label in range(125, 89, -1):

        print("Svm for category- ", label)
        results_file.write("Svm for category #" + str(label+1) + "\n")
        results_file.flush()
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

    results_file.flush()
    # save_svm_data(features, labels, folder_path)

    return errorRates, aucScores


def save_svm_data(features, labels, folder_path):
    try:
        # pickle.dump((features, labels), open('svm.pkl', 'w'))
        print("Trying to pickle svm... ")
        pickle.dump(features, open(folder_path + 'svm-x-data.pkl', 'w'))
        pickle.dump(labels, open(folder_path + 'svm-y-data.pkl', 'w'))
    except Exception as e:
        print(e)
        print(e.message)


if __name__ == '__main__':
    # pickName = "C:/devl/python/ISH_Lasagne/src/DeepLearning/results_dae/learn/run_0/encode.pkl"
    pickled_file = "C:/devl/python/ISH_Lasagne/src/DeepLearning/results_dae/for_debug/run_0/conv_ae.pkl"
    net = pickle.load(open(pickled_file, 'r'))
    run_svm(net)
    errors, aucs = run_svm(net)
    print("Errors", errors)
    print("AUC", aucs)
