# add to kfkd.py
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet

from nolearn.lasagne import TrainSplit
from nolearn.lasagne import BatchIterator


from logistic_sgd import load_data
from  matplotlib import pyplot
import numpy as np


def load2d(test=False, cols=None):
    print 'loading data...'   

    datasets = load_data('ISH.pkl.gz',withZeroMeaning=False)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    
    batch_size = 5
    input_width = 300
    input_height = 140
    train_set_x = train_set_x.reshape(-1, 1, input_width, input_height)
    return train_set_x, train_set_y

net2 = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('conv2', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),
        ('conv3', layers.Conv2DLayer),
        ('pool3', layers.MaxPool2DLayer),
        ('conv4', layers.Conv2DLayer),
        ('pool4', layers.MaxPool2DLayer),
        ('hidden5', layers.DenseLayer),
        ('hidden6', layers.DenseLayer),
        ('hidden7', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    input_shape=(None, 1, 300, 140),
    conv1_num_filters=5, conv1_filter_size=(5, 5), pool1_pool_size=(2, 2),
    conv2_num_filters=10, conv2_filter_size=(9, 9), pool2_pool_size=(2, 2),
    conv3_num_filters=20, conv3_filter_size=(11, 11), pool3_pool_size=(4, 2),
    conv4_num_filters=40, conv4_filter_size=(8, 5), pool4_pool_size=(2, 2),
    hidden5_num_units=500, hidden6_num_units=200, hidden7_num_units=100,
    output_num_units=20, output_nonlinearity=None,

    update_learning_rate=0.1,
    update_momentum=0.9,
    train_split=TrainSplit(eval_size=0.2),
    batch_iterator_train=BatchIterator(batch_size=40),

    regression=True,
    max_epochs=4,
    verbose=1,
    )

X, y = load2d()  # load 2-d data
net2.fit(X, y)

# import numpy as np
# np.sqrt(0.003255) * 48

##############################################
train_loss = np.array([i["train_loss"] for i in net2.train_history_])
valid_loss = np.array([i["valid_loss"] for i in net2.train_history_])
pyplot.plot(train_loss, linewidth=3, label="train")
pyplot.plot(valid_loss, linewidth=3, label="valid")
pyplot.grid()
pyplot.legend()
pyplot.xlabel("epoch")
pyplot.ylabel("loss")
pyplot.ylim(1e-3, 1e-2)
pyplot.yscale("log")
pyplot.show()

#################################################
# def plot_sample(x, y, axis):
#     img = x.reshape(96, 96)
#     axis.imshow(img, cmap='gray')
#     axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=10)
# 
# X, _ = load(test=True)
# y_pred = net1.predict(X)
# 
# fig = pyplot.figure(figsize=(6, 6))
# fig.subplots_adjust(
#     left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
# 
# for i in range(16):
#     ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
#     plot_sample(X[i], y_pred[i], ax)
# 
# pyplot.show()


# Training for 1000 epochs will take a while.  We'll pickle the
# trained model so that we can load it back later:
import cPickle as pickle
with open('net2.pickle', 'wb') as f:
    pickle.dump(net2, f, -1)