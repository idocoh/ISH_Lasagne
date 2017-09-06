from __future__ import print_function
import time
import theano
import platform
from articleCat_CDAE import run
from articleCat_CDAE import load2d
from lasagne.nonlinearities import tanh


def run_all(use_nn_classifier=False, folder_name=None, input_size_pre=None):
    if platform.dist()[0]:
        print ("Running in Ubuntu")
    else:
        print ("Running in Windows")

    print(theano.sandbox.cuda.dnn_available())

    epochs = 20
    num_labels = 15 #164 #TEST: change 15
    amount_train = 16351
    svm_negative_amount = 200
    learning_rate = 0.055

    folder_name = "CAE_" + str(amount_train) + "_noise_rate-" + str(time.time()) if folder_name is None else folder_name

    steps = [
        [5000, 10000, 16352],
        [5000, 10000, 16352],
        [5000, 10000, 16352],
        [5000, 10000, 16352],
        [5000, 11000, 16352],
        [4000, 8000, 12000, 16352],
        [5000, 10000, 15000, 16352],
        [5000, 10000, 16352],
        [2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 16352],
        [2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 16352],
        [5000, 10000, 16352],
        [2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 16352],
        [5000, 10000, 16352],
        [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500,
         9000, 9500, 10000, 10500, 11000, 11500, 12000, 12500, 13000, 13500, 14000, 14500, 15000, 15500,
         16000, 16352],
        [5000, 10000, 16352],
        [5000, 10000, 16352],
        [4000, 8000, 12000, 16352],
        [4000, 8000, 12000, 16352],
        [5000, 10000, 16352],
        [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000, 16352]
    ]
    # indexes   = [0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,  15,  16,  17,  18, 19]
    image_width = [160, 160, 200, 240, 300, 280, 320, 240, 320, 400, 160, 480, 120, 960, 100, 200, 200, 240, 80, 640]
    image_height = [80, 100, 120, 120, 140, 140, 160, 120, 200, 240, 80,  240, 60,  480, 80,  160, 200, 160, 40, 320]
    number_pooling_layers = [2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 1, 4, 2, 2, 2, 2, 1, 4]
    layers_size = [
        [2, 2, 2, 2, 2, 2, 2],
        [4, 4, 4, 4, 4, 4, 4],
        [8, 8, 8, 8, 8, 8, 8],
        [16, 16, 16, 16, 16, 16, 16],
        [16, 32, 32, 64, 32, 32, 16],
        [32, 64, 128, 64, 32]
    ]

    for zero_meaning in [False]:
        try:
            for input_size_index in [16, 11, 3]:
                try:
                    input_size_index = input_size_index if input_size_pre is None else input_size_pre  # for NN test
                    data = load2d(batch_index=1, num_labels=num_labels, TRAIN_PRECENT=1,
                                  steps=steps[input_size_index],
                                  image_width=image_width[input_size_index],
                                  image_height=image_height[input_size_index])

                    learning_rate -= 0.005
                    for num_filters_index in [2]:  # range(0, 3, 1):
                        try:
                            for input_noise_rate in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                                try:
                                    for filter_type in [0]:  # range(2, -1, -2):
                                        try:
                                            for number_conv_layers in [4]:
                                                try:
                                                    for to_shuffle_input in [False]:
                                                        try:
                                                            if input_size_index == 16 and [0, 0.2, 0.5, 0.7, 0.8, 0.9].__contains__(input_size_index):
                                                                continue
                                                            for num_images in range(0, 1, 1):
                                                                # data = load2d(batch_index=1, num_labels=num_labels, TRAIN_PRECENT=1,
                                                                #               steps=steps[input_size_index],
                                                                #               image_width=image_width[input_size_index],
                                                                #               image_height=image_height[input_size_index])
                                                                learning_rate = learning_rate/0.02 if zero_meaning else learning_rate #because std is about 0.02
                                                                filter_type_index = 3 + 2 * filter_type
                                                                print("run number conv layers- ", number_conv_layers)
                                                                print("run Filter type #", filter_type_index)
                                                                print("run Input_noise_rate", input_noise_rate)
                                                                print("run Learning rate- ", learning_rate)
                                                                try:
                                                                    run(layers_size=layers_size[num_filters_index],
                                                                        epochs=epochs,
                                                                        learning_rate=learning_rate,
                                                                        update_momentum=0.9,
                                                                        shuffle_input=to_shuffle_input,
                                                                        number_pooling_layers=number_pooling_layers[input_size_index],
                                                                        dropout_percent=input_noise_rate,
                                                                        loadedData=data,
                                                                        folder_name=folder_name,
                                                                        amount_train=amount_train - num_images*2000,
                                                                        number_conv_layers=number_conv_layers,
                                                                        zero_meaning=zero_meaning,
                                                                        activation=None,
                                                                        last_layer_activation=tanh,
                                                                        filters_type=filter_type_index,
                                                                        train_valid_split=0.001 + 0.002*num_images,
                                                                        input_width=image_width[input_size_index],
                                                                        input_height=image_height[input_size_index],
                                                                        svm_negative_amount=svm_negative_amount,
                                                                        flip_batch=True,
                                                                        batch_size=32,
                                                                        use_nn_classifier=use_nn_classifier)

                                                                except Exception as e:
                                                                    print("failed Filter type #", filter_type_index)
                                                                    print("failed number conv layers- ", number_conv_layers)
                                                                    print("failed Filter number index #", num_filters_index)
                                                                    print("failed Learning rate- ", learning_rate)
                                                                    print(e)
                                                                    print(e.message)
                                                        except Exception as e:
                                                            print(e)
                                                            print(e.message)
                                                except Exception as e:
                                                    print(e)
                                                    print(e.message)
                                        except Exception as e:
                                            print(e)
                                            print(e.message)
                                except Exception as e:
                                    print(e)
                                    print(e.message)
                        except Exception as e:
                            print(e)
                            print(e.message)
                except Exception as e:
                    print(e)
                    print(e.message)
        except Exception as e:
            print(e)
            print(e.message)


if __name__ == "__main__":
    # os.environ["DISPLAY"] = ":99"
    run_all()
