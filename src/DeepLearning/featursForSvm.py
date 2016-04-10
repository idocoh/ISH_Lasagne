import cPickle as pickle
import gzip
import numpy as np
from sklearn import svm
from sklearn.metrics import roc_auc_score
from pickleForArticleCat import pickleAllImages


def images_svm(pickled_file, num_labels=15, TRAIN_SPLIT=0.8):

    FILE_SEPARATOR="/"
    if isinstance(pickled_file, str):
        try:
            features = pickle.load(open(pickled_file, 'r'))
        except:
            f = gzip.open(pickled_file, 'rb')
            features = pickle.load(f)
            f.close()
    else:
        cnn = pickled_file
        pLabel, train_x = pickleAllImages(num_labels=15)
        input_height, input_width, dropout_percent = 300, 140, 0.2
        X_train = train_x.astype(np.float32).reshape((-1, 1, input_height, input_width))
        X_train = np.random.binomial(1, 1-dropout_percent, size=X_train.shape) * X_train
        features = cnn.output_hiddenLayer(X_train)



    if num_labels==15:
        labels_file = "pickled_images"+FILE_SEPARATOR+"articleCat.pkl.gz"
    elif num_labels==20:
        labels_file = "pickled_images"+FILE_SEPARATOR+"topCat.pkl.gz"
    elif num_labels==164:
        labels_file = "pickled_images"+FILE_SEPARATOR+"all164cat.pkl.gz"
    elif num_labels==2081:
        labels_file = "pickled_images"+FILE_SEPARATOR+"all2081cat.pkl.gz"
    else:
        print "bad labels path!!!!!!!"

    print "Loading labels"
    with open(labels_file) as l:
        labels = pickle.load(l)[:features.shape[0], :]
        l.close()

    return features, labels


def checkLabelPredict(features, labels, cross_validation_parts=5):

    positive_data = features[labels == 1, :]
    positive_data = positive_data.reshape((-1, features.shape[-2]*features.shape[-1]))
    negative_data = features[labels == 0, :]
    negative_data = negative_data.reshape((-1, features.shape[-2]*features.shape[-1]))

    if positive_data.shape[0] < cross_validation_parts or negative_data.shape[0] < cross_validation_parts:
        return 0

    negative_data_chunks = np.array_split(negative_data, cross_validation_parts)
    positive_data_chunks = np.array_split(positive_data, cross_validation_parts)

    scores = np.zeros(cross_validation_parts)
    for cross_validation_index in range(0, cross_validation_parts):
        neg_test = negative_data_chunks[cross_validation_index]
        neg_train = np.copy(negative_data_chunks)
        np.delete(neg_train, cross_validation_index)
        neg_train = np.concatenate(neg_train)

        pos_test = positive_data_chunks[cross_validation_index]
        pos_test = generatePositives(pos_test, neg_test.shape[0])
        pos_train = np.copy(positive_data_chunks)
        np.delete(pos_train, cross_validation_index)
        pos_train = np.concatenate(pos_train)
        pos_train = generatePositives(pos_train, neg_train.shape[0])

        clf = svm.SVC(kernel='linear', C=1).fit(np.concatenate((pos_train, neg_train), axis=0),
                                                np.concatenate((np.ones(pos_train.shape[0]),
                                                                np.zeros(neg_train.shape[0])), axis=0))
        score = clf.score(np.concatenate((pos_test, neg_test), axis=0),
                          np.concatenate((np.ones(pos_test.shape[0]), np.zeros(neg_test.shape[0])), axis=0))
        # auc_score = roc_auc_score(test_y, test_predict)

        scores[cross_validation_index] = score
        print score

    return np.average(scores)


def generatePositives(positives, num_negatives):
    num_positives = positives.shape[0]
    multiple_by = np.ones(num_positives)*np.divide(num_negatives, num_positives)
    for i in range(0, np.remainder(num_negatives, num_positives)):
        multiple_by[i] += 1

    return np.repeat(positives, multiple_by.astype(int), axis=0)

def run_svm(pickName):
    num_labels = 15
    features, labels = images_svm(pickName, num_labels=num_labels)
    errorRates = np.zeros(num_labels)
    # aucScores = np.zeros(num_labels)
    for label in range(0, num_labels):
        print "Svm for category- ", label
        labelPredRate = checkLabelPredict(features, labels[:, label])
        errorRates[label] = labelPredRate
        # aucScores[label] = labelAucScore

    errorRate = np.average(errorRates)
    # aucAverageScore = np.average(aucScores)
    print "Average error- ", errorRate, "%"
    print "Prediction rate- ", 100 - errorRate, "%"
    # print "Average Auc Score- ", aucAverageScore

    # return (errorRates, aucScores)

if __name__ == '__main__':
    pickName = "C:/devl/python/ISH_Lasagne/src/DeepLearning/results_dae/learn/run_0/encode.pkl"
    # pickled_file = "C:/devl/python/ISH_Lasagne/src/DeepLearning/results_dae/CAE_3000_2Conv2Pool-1460319214/run_0/conv_ae.pkl"
    # net = pickle.load(open(pickled_file, 'r'))
    # pickName.load_weights_from(net)

    run_svm(pickName)
