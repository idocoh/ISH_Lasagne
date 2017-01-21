from __future__ import print_function
import time
from lasagne.nonlinearities import tanh, LeakyRectify
from lasagne import layers
from lasagne.updates import nesterov_momentum
from PIL import Image
from featursForSvm import run_svm
import cPickle as pickle
import platform

from logistic_sgd import load_data
from nolearn.lasagne import BatchIterator
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import PrintLayerInfo
from nolearn.lasagne import TrainSplit
import numpy as np
from shape import ReshapeLayer
import theano.sandbox.cuda


FILE_SEPARATOR = "/"
counter = 0


def load2d(num_labels, batch_index=1, outputFile=None, input_width=300, input_height=140, end_index=16351, MULTI_POSITIVES=20,
           dropout_percent=0.1, data_set='ISH.pkl.gz', toShuffleInput = False, withZeroMeaning = False, TRAIN_PRECENT=0.8):
    print ('loading data...')

    data_sets = load_data(data_set, batch_index=batch_index, withSVM=0, toShuffleInput=toShuffleInput,
                                               withZeroMeaning=withZeroMeaning,
                                               MULTI_POSITIVES=MULTI_POSITIVES, dropout_percent=dropout_percent,
                                               labelset=num_labels, TRAIN_DATA_PRECENT=TRAIN_PRECENT)

    train_set_x, train_set_y = data_sets[0]
#     valid_set_x, valid_set_y = data_sets[1]
    test_set_x, test_set_y = data_sets[2]
    print(train_set_x.shape[0], ' samples loaded')
    return (train_set_x, train_set_y, test_set_x, test_set_y)


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


def run(loadedData=None, learning_rate=0.04, update_momentum=0.9, update_rho=None, epochs=15,
        input_width=300, input_height=140, train_valid_split=0.2, multiple_positives=20, flip_batch=True,
        dropout_percent=0.1, amount_train=16351, activation=None, last_layer_activation=None, batch_size=32,
        layers_size=[5, 10, 20, 40], shuffle_input=False, zero_meaning=False, filters_type=3,
        input_noise_rate=0.3, pre_train_epochs=1, softmax_train_epochs=2, fine_tune_epochs=2,
        categories=15, svm_negative_amount=800, folder_name="default", dataset='withOutDataSet'):

    global counter
    folder_path = "results_dae"+FILE_SEPARATOR + folder_name + FILE_SEPARATOR + "run_" + str(counter) + FILE_SEPARATOR
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    PARAMS_FILE_NAME = folder_path + "parameters.txt"
    HIDDEN_LAYER_OUTPUT_FILE_NAME = folder_path + "hiddenLayerOutput.pkl.gz"
    FIG_FILE_NAME = folder_path + "fig"
    PICKLES_NET_FILE_NAME = folder_path + "picklesNN.pkl.gz"
    SVM_FILE_NAME = folder_path + "svmData.txt"
    LOG_FILE_NAME = folder_path + "message.log"

    All_Results_FIle = "results_dae"+FILE_SEPARATOR + "all_results.txt"


    #     old_stdout = sys.stdout
    #     print "less",LOG_FILE_NAME
    log_file = False  #open(LOG_FILE_NAME, "w")
    #     sys.stdout = log_file

    counter += 1
    output_file = open(PARAMS_FILE_NAME, "w")
    results_file = open(All_Results_FIle, "a")

    if filters_type == 3:
        filter_1 = (3, 3)
        filter_2 = (3, 3)
        filter_3 = (3, 3)
        filter_4 = (3, 3)
        filter_5 = (3, 3)
        filter_6 = (3, 3)
    elif filters_type == 5:
        filter_1 = (5, 5)
        filter_2 = (5, 5)
        filter_3 = (5, 5)
        filter_4 = (5, 5)
        filter_5 = (5, 5)
        filter_6 = (5, 5)
    elif filters_type == 7:
        filter_1 = (7, 7)
        filter_2 = (7, 7)
        filter_3 = (5, 5)
        filter_4 = (7, 7)
        filter_5 = (7, 7)
        filter_6 = (5, 5)
    elif filters_type == 9:
        filter_1 = (9, 9)
        filter_2 = (7, 7)
        filter_3 = (5, 5)
        filter_4 = (7, 7)
        filter_5 = (9, 9)
        filter_6 = (5, 5)
    elif filters_type == 11:
        filter_1 = (11, 11)
        filter_2 = (9, 9)
        filter_3 = (7, 7)
        filter_4 = (9, 9)
        filter_5 = (11, 11)
        filter_6 = (7, 7)

    def create_cae(input_height, input_width):

        cnn = NeuralNet(layers=[
            ('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),
            ('conv11', layers.Conv2DLayer),
            ('conv12', layers.Conv2DLayer),
            ('conv13', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),
            ('conv2', layers.Conv2DLayer),
            ('conv21', layers.Conv2DLayer),
            ('conv22', layers.Conv2DLayer),
            ('conv23', layers.Conv2DLayer),
            ('pool2', layers.MaxPool2DLayer),
            ('conv3', layers.Conv2DLayer),
            ('conv31', layers.Conv2DLayer),
            ('conv32', layers.Conv2DLayer),
            ('conv33', layers.Conv2DLayer),
            ('unpool1', Unpool2DLayer),
            ('conv4', layers.Conv2DLayer),
            ('conv41', layers.Conv2DLayer),
            ('conv42', layers.Conv2DLayer),
            ('conv43', layers.Conv2DLayer),
            ('unpool2', Unpool2DLayer),
            ('conv5', layers.Conv2DLayer),
            ('conv51', layers.Conv2DLayer),
            ('conv52', layers.Conv2DLayer),
            ('conv53', layers.Conv2DLayer),
            ('conv6', layers.Conv2DLayer),
            ('output_layer', ReshapeLayer),
        ],

            input_shape=(None, 1, input_width, input_height),
            # Layer current size - 1x300x140

            conv1_num_filters=layers_size[0], conv1_filter_size=filter_1, conv1_nonlinearity=activation,
            # conv1_border_mode="same",
            conv1_pad="same",
            conv11_num_filters=layers_size[0], conv11_filter_size=filter_1, conv11_nonlinearity=activation,
            # conv11_border_mode="same",
            conv11_pad="same",
            conv12_num_filters=layers_size[0], conv12_filter_size=filter_1, conv12_nonlinearity=activation,
            # conv12_border_mode="same",
            conv12_pad="same",
            conv13_num_filters=layers_size[0], conv13_filter_size=filter_1, conv13_nonlinearity=activation,
            # conv13_border_mode="same",
            conv13_pad="same",

            pool1_pool_size=(2, 2),

            conv2_num_filters=layers_size[1], conv2_filter_size=filter_2, conv2_nonlinearity=activation,
            # conv2_border_mode="same",
            conv2_pad="same",
            conv21_num_filters=layers_size[1], conv21_filter_size=filter_2, conv21_nonlinearity=activation,
            # conv21_border_mode="same",
            conv21_pad="same",
            conv22_num_filters=layers_size[1], conv22_filter_size=filter_2, conv22_nonlinearity=activation,
            # conv22_border_mode="same",
            conv22_pad="same",
            conv23_num_filters=layers_size[1], conv23_filter_size=filter_2, conv23_nonlinearity=activation,
            # conv23_border_mode="same",
            conv23_pad="same",

            pool2_pool_size=(2, 2),

            conv3_num_filters=layers_size[2], conv3_filter_size=filter_3, conv3_nonlinearity=activation,
            # conv3_border_mode="same",
            conv3_pad="same",
            conv31_num_filters=layers_size[2], conv31_filter_size=filter_3, conv31_nonlinearity=activation,
            # conv31_border_mode="same",
            conv31_pad="same",
            conv32_num_filters=layers_size[2], conv32_filter_size=filter_3, conv32_nonlinearity=activation,
            # conv32_border_mode="same",
            conv32_pad="same",
            conv33_num_filters=1, conv33_filter_size=filter_3, conv33_nonlinearity=activation,
            # conv33_border_mode="same",
            conv33_pad="same",

            unpool1_ds=(2, 2),

            conv4_num_filters=layers_size[3], conv4_filter_size=filter_4, conv4_nonlinearity=activation,
            # conv4_border_mode="same",
            conv4_pad="same",
            conv41_num_filters=layers_size[3], conv41_filter_size=filter_4, conv41_nonlinearity=activation,
            # conv41_border_mode="same",
            conv41_pad="same",
            conv42_num_filters=layers_size[3], conv42_filter_size=filter_4, conv42_nonlinearity=activation,
            # conv42_border_mode="same",
            conv42_pad="same",
            conv43_num_filters=layers_size[3], conv43_filter_size=filter_4, conv43_nonlinearity=activation,
            # conv43_border_mode="same",
            conv43_pad="same",

            unpool2_ds=(2, 2),

            conv5_num_filters=layers_size[4], conv5_filter_size=filter_5, conv5_nonlinearity=activation,
            # conv5_border_mode="same",
            conv5_pad="same",
            conv51_num_filters=layers_size[4], conv51_filter_size=filter_5, conv51_nonlinearity=activation,
            # conv51_border_mode="same",
            conv51_pad="same",
            conv52_num_filters=layers_size[4], conv52_filter_size=filter_5, conv52_nonlinearity=activation,
            # conv52_border_mode="same",
            conv52_pad="same",
            conv53_num_filters=layers_size[4], conv53_filter_size=filter_5, conv53_nonlinearity=activation,
            # conv53_border_mode="same",
            conv53_pad="same",

            conv6_num_filters=1, conv6_filter_size=filter_6, conv6_nonlinearity=last_layer_activation,
            # conv6_border_mode="same",
            conv6_pad="same",

            output_layer_shape=(([0], -1)),

            update_learning_rate=learning_rate,
            update_momentum=update_momentum,
            update=nesterov_momentum,
            train_split=TrainSplit(eval_size=train_valid_split),
            batch_iterator_train=FlipBatchIterator(batch_size=batch_size) if flip_batch else BatchIterator(batch_size=batch_size),
            regression=True,
            max_epochs=epochs,
            verbose=1,
            hiddenLayer_to_output=14)

        return cnn

    def train_cae(cnn, input_height, input_width, X_train, X_out):

        X_train *= np.random.binomial(1, 1-dropout_percent, size=X_train.shape)
        print('Training CAE with ', X_train.shape[0], ' samples')
        cnn.fit(X_train, X_out)

        # try:
        #     pickle.dump(cnn, open(folder_path + 'conv_ae.pkl', 'w'))
        #     # cnn = pickle.load(open(folder_path + 'conv_ae.pkl','r'))
        #     cnn.save_weights_to(folder_path + 'conv_ae.np')
        # except:
        #     print ("Could not pickle cnn")

        print('Predicting ', X_train.shape[0], ' samples through CAE')
        # X_pred = cnn.predict(X_train).reshape(-1, input_height, input_width)  # * sigma + mu

        # # X_pred = np.rint(X_pred).astype(int)
        # # X_pred = np.clip(X_pred, a_min=0, a_max=255)
        # # X_pred = X_pred.astype('uint8')
        #
        # try:
        #     trian_last_hiddenLayer = cnn.output_hiddenLayer(X_train)
        #     # test_last_hiddenLayer = cnn.output_hiddenLayer(test_x)
        #     pickle.dump(trian_last_hiddenLayer, open(folder_path + 'encode.pkl', 'w'))
        # except:
        #     print "Could not save encoded images"

        # save_example_images(X_out, X_pred, X_train)

        return cnn

    def save_example_images(X_out, X_pred, X_train):
        print("Saving some images....")
        for i in range(10):
            index = np.random.randint(X_train.shape[0])
            print(index)

            def get_picture_array(X, index):
                array = np.rint(X[index] * 256).astype(np.int).reshape(input_height, input_width)
                array = np.clip(array, a_min=0, a_max=255)
                return array.repeat(4, axis=0).repeat(4, axis=1).astype(np.uint8())

            original_image = Image.fromarray(get_picture_array(X_out, index))
            # original_image.save(folder_path + 'original' + str(index) + '.png', format="PNG")
            #
            # array = np.rint(trian_last_hiddenLayer[index] * 256).astype(np.int).reshape(input_height/2, input_width/2)
            # array = np.clip(array, a_min=0, a_max=255)
            # encode_image = Image.fromarray(array.repeat(4, axis=0).repeat(4, axis=1).astype(np.uint8()))
            # encode_image.save(folder_path + 'encode' + str(index) + '.png', format="PNG")

            new_size = (original_image.size[0] * 3, original_image.size[1])
            new_im = Image.new('L', new_size)
            new_im.paste(original_image, (0, 0))
            pred_image = Image.fromarray(get_picture_array(X_pred, index))
            # pred_image.save(folder_path + 'pred' + str(index) + '.png', format="PNG")
            new_im.paste(pred_image, (original_image.size[0], 0))

            noise_image = Image.fromarray(get_picture_array(X_train, index))
            new_im.paste(noise_image, (original_image.size[0] * 2, 0))
            new_im.save(folder_path + 'origin_prediction_noise-' + str(index) + '.png', format="PNG")

            # diff = ImageChops.difference(original_image, pred_image)
            # diff = diff.convert('L')
            # diff.save(folder_path + 'diff' + str(index) + '.png', format="PNG")

            # plt.imshow(new_im)
            # new_size = (original_image.size[0] * 2, original_image.size[1])
            # new_im = Image.new('L', new_size)
            # new_im.paste(original_image, (0, 0))
            # pred_image = Image.fromarray(get_picture_array(X_train, index))
            # # pred_image.save(folder_path + 'noisyInput' + str(index) + '.png', format="PNG")
            # new_im.paste(pred_image, (original_image.size[0], 0))
            # new_im.save(folder_path+'origin_VS_noise-'+str(index)+'.png', format="PNG")
            # plt.imshow(new_im)

    def write_output_file(output_file, train_history, layer_info):
        # save the network's parameters
        output_file.write("Validation set error: " + str(train_history[-1]['valid_accuracy']) + "\n\n")
        results_file.write(str(train_history[-1]['valid_accuracy']) + "\t")

        output_file.write("Training NN on: " + ("20 Top Categories\n" if 20 == categories else "Article Categories\n"))
        output_file.write("Learning rate: " + str(learning_rate) + "\n")
        results_file.write(str(learning_rate) + "\t")
        output_file.write(("Momentum: " + str(update_momentum) + "\n") if update_rho is None else (
            "Decay Factor: " + str(update_rho) + "\n"))
        results_file.write(str(update_momentum) + "\t")
        output_file.write(("FlipBatcherIterater" if flip_batch else "BatchIterator") + " with batch: " + str(batch_size) + "\n")
        results_file.write("FlipBatcherIterater\t" + str(batch_size) + "\t")
        output_file.write("Num epochs: " + str(epochs) + "\n")
        results_file.write(str(epochs) + "\t")
        output_file.write("Layers size: " + str(layers_size) + "\n\n")
        results_file.write(str(layers_size) + "\t")
        output_file.write("Activation func: " + ("Rectify" if activation is None else str(activation)) + "\n")
        results_file.write(("Rectify" if activation is None else str(activation)) + "\t")
        output_file.write(
            "Last layer activation func: " + ("Rectify" if last_layer_activation is None else str(last_layer_activation)) + "\n")
        results_file.write(("Rectify" if last_layer_activation is None else str(last_layer_activation)) + "\t")
        #         output_file.write("Multiple Positives by: " + str(multiple_positives) + "\n")
        output_file.write("Number of images for training: " + str(amount_train) + "\n")
        results_file.write(str(amount_train) + "\t")
        output_file.write("Number of negative images in svm: " + str(svm_negative_amount) + "\n")
        output_file.write("Dropout noise precent: " + str(dropout_percent * 100) + "%\n")
        results_file.write(str(dropout_percent * 100) + "%\t")
        output_file.write("Train/validation split: " + str(train_valid_split) + "\n")
        results_file.write(str(train_valid_split) + "\t")
        output_file.write("shuffle_input: " + str(shuffle_input) + "\n")
        results_file.write(str(shuffle_input) + "\t")
        output_file.write("zero_meaning: " + str(zero_meaning) + "\n\n")
        results_file.write(str(zero_meaning) + "\t")

        output_file.write("history: " + str(train_history) + "\n\n")
        results_file.write(str(train_history) + "\t")
        output_file.write("layer_info:\n" + str(layer_info) + "\n")
        results_file.write("[" + str(layer_info).replace("\n", ",") + "]\t")
        output_file.write("filters_info:\n" + str(filter_1) + "\n")
        output_file.write(str(filter_2) + "\n")
        output_file.write(str(filter_3) + "\n")
        output_file.write(str(filter_4) + "\n")
        output_file.write(str(filter_5) + "\n")
        output_file.write(str(filter_6) + "\n\n")
        results_file.write("{" + str((filter_1, filter_2, filter_3, filter_4, filter_5, filter_6)) + "]\t")
        output_file.write("Run time[minutes] is: " + str(run_time) + "\n")

        output_file.flush()
        results_file.write(str(time.ctime()) + "\t")
        results_file.write(folder_name + "\n")
        results_file.flush()

    start_time = time.clock()
    print ("Start time: ", time.ctime())

    if loadedData is None:
        train_x, train_y, test_x, test_y = load2d(categories, output_file, input_width, input_height, amount_train, multiple_positives, dropout_percent)  # load 2-d data
    else:
        data = loadedData
        train_x, train_y, test_x, test_y = data
    cnn = create_cae(input_height, input_width)

    if zero_meaning:
        train_x = train_x.astype(np.float64)
        mu, sigma = np.mean(train_x.flatten()), np.std(train_x.flatten())
        print("Mean- ", mu)
        print("Std- ", sigma)
        train_x = (train_x - mu) / sigma

    x_train = train_x.astype(np.float32).reshape((-1, 1, input_width, input_height))
    x_flat = x_train.reshape((x_train.shape[0], -1))
    cnn = train_cae(cnn, input_height, input_width, x_train[:amount_train], x_flat[:amount_train])

    run_time = (time.clock() - start_time) / 60.
    write_output_file(output_file, cnn.train_history_, PrintLayerInfo._get_layer_info_plain(cnn))
    print ("Learning took (min)- ", run_time)

    valid_accuracy = cnn.train_history_[-1]['valid_accuracy']
    if valid_accuracy > 0.05:
        return valid_accuracy

    try:
        print("Running SVM")
        errors, aucs = run_svm(cnn, X_train=x_train, labels=train_y, svm_negative_amount=svm_negative_amount)
        print("Errors", errors)
        print("AUC", aucs)
        output_file.write("SVM errors: " + str(errors))
        output_file.write("SVM auc: " + str(aucs))
        results_file.write(str(aucs) + "\n")

        output_file.flush()
        results_file.flush()
    except Exception as e:
        print(e)
        print(e.message)

    return valid_accuracy


def run_all():
    if platform.dist()[0]:
        print ("Running in Ubuntu")
    else:
        print ("Running in Windows")

    print(theano.sandbox.cuda.dnn_available())

    num_labels = 15
    amount_train = 30
    svm_negative_amount = 30
    input_noise_rate = 0.2
    zero_meaning = False
    epochs = 2
    folder_name = "CAE_" + str(amount_train) + "_3Conv2Pool9Filters_different3000Batch1-"+str(time.time())
    data = load2d(batch_index=1, num_labels=num_labels, TRAIN_PRECENT=1)

    for i in range(1, 2, 1):
        print("Run #", i)
        try:
            run(layers_size=[32, 32, 64, 32, 32], epochs=epochs, learning_rate=0.054+0.0005*i, update_momentum=0.9,
                dropout_percent=input_noise_rate, loadedData=data, folder_name=folder_name, amount_train=amount_train,
                zero_meaning=zero_meaning, activation=None, last_layer_activation=tanh, filters_type=11,
                svm_negative_amount=svm_negative_amount)

        except Exception as e:
            print("failed to run- ", i)
            print(e)

if __name__ == "__main__":
    import os
    os.environ["DISPLAY"] = ":99"
    run_all()
