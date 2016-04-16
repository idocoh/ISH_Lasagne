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

# <codecell>

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


def images_svm(pickled_file, x=None, all_labels=None, num_labels=15, TRAIN_SPLIT=0.8):

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
        if x is None:
            all_labels, train_x = pickleAllImages(num_labels=15)
            input_height, input_width, dropout_percent = 300, 140, 0.2
            x = train_x.astype(np.float32).reshape((-1, 1, input_height, input_width))
            x *= np.random.binomial(1, 1 - dropout_percent, size=x.shape)
        # features = cnn.output_hiddenLayer(x)

        quarter_x = np.floor(x.shape[0] / 4)

        train_last_hidden_layer_1 = cnn.output_hiddenLayer(x[:quarter_x])
        print("after first quarter train output")
        train_last_hidden_layer_2 = cnn.output_hiddenLayer(x[quarter_x:2 * quarter_x])
        print("after second quarter train output")
        train_last_hidden_layer_3 = cnn.output_hiddenLayer(x[2 * quarter_x: 3 * quarter_x])
        print("after third quarter train output")
        train_last_hidden_layer_4 = cnn.output_hiddenLayer(x[3 * quarter_x:])

        features = np.concatenate((train_last_hidden_layer_1, train_last_hidden_layer_2, train_last_hidden_layer_3, train_last_hidden_layer_4), axis=0)
        print('Features size: ', features.shape)
        print('Labels size: ', all_labels.shape)
        # features = pickled_file



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


def checkLabelPredict(features, labels, cross_validation_parts=5):
    try:
        print('Inside check label:')
        print (features.shape)
        print (labels.shape)
        print((labels==1).shape)
        print ((features[labels == 1, :]).shape)
    except:
        pass
    positive_data = features[labels == 1, :]
    positive_data = positive_data.reshape((-1, features.shape[-2]*features.shape[-1]))
    negative_data = features[labels == 0, :]
    negative_data = negative_data.reshape((-1, features.shape[-2]*features.shape[-1]))

    if positive_data.shape[0] < cross_validation_parts or negative_data.shape[0] < cross_validation_parts:
        return 1

    negative_data_chunks = np.array_split(negative_data, cross_validation_parts)
    positive_data_chunks = np.array_split(positive_data, cross_validation_parts)

    scores = np.zeros(cross_validation_parts)
    for cross_validation_index in range(0, cross_validation_parts):
        neg_test = negative_data_chunks[cross_validation_index]
        neg_train = np.copy(negative_data_chunks)
        np.delete(neg_train, cross_validation_index)
        neg_train = np.concatenate(neg_train)

        pos_test = positive_data_chunks[cross_validation_index]
        pos_test = generate_positives(pos_test, neg_test.shape[0])
        pos_train = np.copy(positive_data_chunks)
        np.delete(pos_train, cross_validation_index)
        pos_train = np.concatenate(pos_train)
        pos_train = generate_positives(pos_train, neg_train.shape[0])

        clf = svm.SVC(kernel='linear', C=1).fit(np.concatenate((pos_train, neg_train), axis=0),
                                                np.concatenate((np.ones(pos_train.shape[0]),
                                                                np.zeros(neg_train.shape[0])), axis=0))
        score = clf.score(np.concatenate((pos_test, neg_test), axis=0),
                          np.concatenate((np.ones(pos_test.shape[0]), np.zeros(neg_test.shape[0])), axis=0))
        # auc_score = roc_auc_score(test_y, test_predict)

        scores[cross_validation_index] = score
        print(score)

    return np.average(scores)


def generate_positives(positives, num_negatives):
    num_positives = positives.shape[0]
    multiple_by = np.ones(num_positives)*np.divide(num_negatives, num_positives)
    for i in range(0, np.remainder(num_negatives, num_positives)):
        multiple_by[i] += 1

    return np.repeat(positives, multiple_by.astype(int), axis=0)


def run_svm(pickle_name, X_train=None, labels=None):
    num_labels = 15
    features, labels = images_svm(pickle_name, X_train, labels,  num_labels=num_labels)
    errorRates = np.zeros(num_labels)
    # aucScores = np.zeros(num_labels)

    for label in range(0, num_labels):
        print("Svm for category- ", label)
        labelPredRate = checkLabelPredict(features, labels[:, label])
        errorRates[label] = labelPredRate
        # aucScores[label] = labelAucScore

    errorRate = np.average(errorRates)
    # aucAverageScore = np.average(aucScores)
    print("Average error- ", errorRate, "%")
    print("Prediction rate- ", 100 - errorRate, "%")
    # print "Average Auc Score- ", aucAverageScore

    # return (errorRates, aucScores)
    return errorRates

if __name__ == '__main__':
    # pickName = "C:/devl/python/ISH_Lasagne/src/DeepLearning/results_dae/learn/run_0/encode.pkl"
    pickled_file = "C:/devl/python/ISH_Lasagne/src/DeepLearning/results_dae/CAE_3000_2Conv2Pool-1460319214/run_0/conv_ae.pkl"
    net = pickle.load(open(pickled_file, 'r'))
    run_svm(net)

    # pLabel, train_x = pickleAllImages(num_labels=15)
    # input_height, input_width, dropout_percent = 300, 140, 0.2
    # X_train = train_x.astype(np.float32).reshape((-1, 1, input_height, input_width))
    # X_train *= np.random.binomial(1, 1 - dropout_percent, size=X_train.shape)

    config.experimental.unpickle_gpu_on_cpu = True

    pkl_cnn_folder = "C:/devl/python/ISH_Lasagne/src/DeepLearning/pklCNN/"

    for dirname, dirnames, filenames in os.walk(pkl_cnn_folder):
        # print path to all subdirectories first.
        print("Running in folder: ", dirname)
        # for subdirname in dirnames:
        #     print('hello', os.path.join(dirname, subdirname))

        # print path to all filenames.
        for filename in filenames:
            if 'conv_ae.pkl' in filename:
                try:
                    file_name = os.path.join(dirname, filename).replace('\\', '/')
                    print('SVM for: ', file_name)
                    cnn = pickle.load(open(file_name, 'r'))
                    # run_svm(cnn, X_train, pLabel)
                    print('Success')
                except Exception as e:
                    print(e)
                # # Advanced usage:
                # # editing the 'dirnames' list will stop os.walk() from recursing into there.
                # if '.git' in dirnames:
                #     # don't go into any .git directories.
                #     dirnames.remove('.git')


