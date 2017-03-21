from lasagne import layers
from lasagne.updates import nesterov_momentum
from sklearn.metrics import roc_auc_score
from nolearn.lasagne import NeuralNet
import lasagne

from logistic_sgd import load_data
from nolearn.lasagne import BatchIterator
from nolearn.lasagne import TrainSplit
import numpy as np
import cPickle as pickle
import gzip


def runNNclassifier(train_params, train_y, test_params, test_y, LEARNING_RATE=0.04, UPDATE_MOMENTUM=0.9, UPDATE_RHO=None,
                    NUM_OF_EPOCH=10, TRAIN_VALIDATION_SPLIT=0.2, NUM_UNITS_HIDDEN_LAYER=[1000, 100], BATCH_SIZE=32):

    def getNN(pickledFilePath=None):
        if pickledFilePath is not None: #isinstance(pickledFilePath, str):
            try:
                with open(pickledFilePath, 'rb') as f:
                    classifier_net = pickle.load(f)
                    f.close()
            except :
                f = gzip.open(pickledFilePath, 'rb')
                classifier_net = pickle.load(f)
                f.close()
        else:
            if len(NUM_UNITS_HIDDEN_LAYER) == 2:
                classifier_net = NeuralNet(layers=[
                            ('input', layers.InputLayer),
                            ('hidden1', layers.DenseLayer),
                            ('hidden2', layers.DenseLayer),
                            ('output', layers.DenseLayer)],
                        input_shape=(None, input_length),
                        hidden1_num_units=NUM_UNITS_HIDDEN_LAYER[0], hidden2_num_units=NUM_UNITS_HIDDEN_LAYER[1],
                        output_num_units=2,
                        output_nonlinearity=lasagne.nonlinearities.softmax,
                        update_learning_rate=LEARNING_RATE,
                        update_momentum=UPDATE_MOMENTUM,
                        update=nesterov_momentum,
                        train_split=TrainSplit(eval_size=TRAIN_VALIDATION_SPLIT),
                        batch_iterator_train=BatchIterator(batch_size=BATCH_SIZE),
                        regression=True,
                        max_epochs=NUM_OF_EPOCH,
                        verbose=1,
                        hiddenLayer_to_output=-2)

            elif len(NUM_UNITS_HIDDEN_LAYER) == 3:
                classifier_net = NeuralNet(layers=[
                            ('input', layers.InputLayer),
                            ('hidden1', layers.DenseLayer),
                            ('hidden2', layers.DenseLayer),
                            ('hidden3', layers.DenseLayer),
                            ('output', layers.DenseLayer)],
                        input_shape=(None, input_length),
                        hidden1_num_units=NUM_UNITS_HIDDEN_LAYER[0],
                        hidden2_num_units=NUM_UNITS_HIDDEN_LAYER[1],
                        hidden3_num_units=NUM_UNITS_HIDDEN_LAYER[2],
                        output_num_units=2,
                        output_nonlinearity=lasagne.nonlinearities.softmax,
                        update_learning_rate=LEARNING_RATE,
                        update_momentum=UPDATE_MOMENTUM,
                        update=nesterov_momentum,
                        train_split=TrainSplit(eval_size=TRAIN_VALIDATION_SPLIT),
                        batch_iterator_train=BatchIterator(batch_size=BATCH_SIZE),
                        regression=True,
                        max_epochs=NUM_OF_EPOCH,
                        verbose=1,
                        hiddenLayer_to_output=-2)
            elif len(NUM_UNITS_HIDDEN_LAYER) == 4:
                classifier_net = NeuralNet(layers=[
                    ('input', layers.InputLayer),
                    ('hidden1', layers.DenseLayer),
                    ('hidden2', layers.DenseLayer),
                    ('hidden3', layers.DenseLayer),
                    ('hidden4', layers.DenseLayer),
                    ('output', layers.DenseLayer)],
                    input_shape=(None, input_length),
                    hidden1_num_units=NUM_UNITS_HIDDEN_LAYER[0],
                    hidden2_num_units=NUM_UNITS_HIDDEN_LAYER[1],
                    hidden3_num_units=NUM_UNITS_HIDDEN_LAYER[2],
                    hidden4_num_units=NUM_UNITS_HIDDEN_LAYER[3],
                    output_num_units=2,
                    output_nonlinearity=lasagne.nonlinearities.softmax,
                    update_learning_rate=LEARNING_RATE,
                    update_momentum=UPDATE_MOMENTUM,
                    update=nesterov_momentum,
                    train_split=TrainSplit(eval_size=TRAIN_VALIDATION_SPLIT),
                    batch_iterator_train=BatchIterator(batch_size=BATCH_SIZE),
                    regression=True,
                    max_epochs=NUM_OF_EPOCH,
                    verbose=1,
                    hiddenLayer_to_output=-2)
            elif len(NUM_UNITS_HIDDEN_LAYER) == 5:
                classifier_net = NeuralNet(layers=[
                    ('input', layers.InputLayer),
                    ('hidden1', layers.DenseLayer),
                    ('hidden2', layers.DenseLayer),
                    ('hidden3', layers.DenseLayer),
                    ('hidden4', layers.DenseLayer),
                    ('hidden4', layers.DenseLayer),
                    ('output', layers.DenseLayer)],
                    input_shape=(None, input_length),
                    hidden1_num_units=NUM_UNITS_HIDDEN_LAYER[0],
                    hidden2_num_units=NUM_UNITS_HIDDEN_LAYER[1],
                    hidden3_num_units=NUM_UNITS_HIDDEN_LAYER[2],
                    hidden4_num_units=NUM_UNITS_HIDDEN_LAYER[3],
                    hidden5_num_units=NUM_UNITS_HIDDEN_LAYER[4],
                    output_num_units=2,
                    output_nonlinearity=lasagne.nonlinearities.softmax,
                    update_learning_rate=LEARNING_RATE,
                    update_momentum=UPDATE_MOMENTUM,
                    update=nesterov_momentum,
                    train_split=TrainSplit(eval_size=TRAIN_VALIDATION_SPLIT),
                    batch_iterator_train=BatchIterator(batch_size=BATCH_SIZE),
                    regression=True,
                    max_epochs=NUM_OF_EPOCH,
                    verbose=1,
                    hiddenLayer_to_output=-2)

        return classifier_net

    input_length = train_params.shape[1]
    classifier_net = getNN()

    classifier_net.fit(train_params, train_y)
    test_predict = classifier_net.predict(test_params)

    # look only on the positive side
    test_predict = test_predict[:, 0]

    differ = test_predict-test_y[:, 0]
    error_rate = np.sum(np.abs(differ))/differ.shape[0] * 100
    auc_score = roc_auc_score(test_y[:, 0], test_predict)

    # print differ
    print "        Error- ", error_rate, "%"
    print "            RocAucScore- ", auc_score
    return classifier_net, error_rate, auc_score



if __name__ == "__main__":
    main_dir = "C:\\devl\\python\\temp\\"
    pickName = main_dir
    runNNclassifier(FOLDER_NAME=pickName, NUM_CAT=15)
