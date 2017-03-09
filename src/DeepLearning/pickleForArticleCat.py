import cPickle
import gzip
import numpy as np


def pickleAllImages(num_labels, end_index=16351, svm_size=600, steps=[5000, 10000, 15000, 16352], image_width=320, image_height=160):

    FILE_SEPARATOR="/"

    if num_labels == 15:
        labelsFile = "pickled_images"+FILE_SEPARATOR+"articleCat.pkl.gz"
    elif num_labels == 20:
        labelsFile = "pickled_images"+FILE_SEPARATOR+"topCat.pkl.gz"
    elif num_labels == 164:
        labelsFile = "pickled_images"+FILE_SEPARATOR+"all164cat.pkl.gz"
    elif num_labels == 2081:
        labelsFile = "pickled_images"+FILE_SEPARATOR+"all2081cat.pkl.gz"
    else:
        print "bad label path!!!!!!!"
         
    print "Loading labels..."
    with open(labelsFile) as l:
        pLabel = cPickle.load(l)
        l.close()
        print "     Done"

    # dir1 = "pickled_images"+FILE_SEPARATOR+"ISH-noLearn_0_5000_320_160.pkl.gz"#300_140.pkl.gz"
    dir1 = "pickled_images"+FILE_SEPARATOR+"ISH-noLearn_0_"+str(steps[0])+"_"+str(image_width)+"_"+str(image_height)+".pkl.gz"

    f1 = gzip.open(dir1, 'rb')
    train_set1, valid_set1, test_set1 = cPickle.load(f1)
    f1.close()
    print "after reading part 1"

    if end_index <= steps[0]:
        pData = train_set1[0]
        print "Done Reading images"
        if svm_size > 0:
            return seperateSVM(pData, pLabel, svm_size)
        else:
            return pLabel[:end_index], pData[:end_index]

    #dir2 = "pickled_images"+FILE_SEPARATOR+"ISH-noLearn_5000_10000_320_160.pkl.gz"#300_140.pkl.gz"
    dir2 = "pickled_images"+FILE_SEPARATOR+"ISH-noLearn_"+str(steps[0])+"_"+str(steps[1])+"_"+str(image_width)+"_"+str(image_height)+".pkl.gz"

    f2 = gzip.open(dir2, 'rb')
    train_set2, valid_set2, test_set2 = cPickle.load(f2)
    f2.close()
    print "after reading part 2"

    if end_index <= steps[1]:
        pData = np.concatenate((train_set1[0], train_set2[0]), axis=0)
        print "Done Reading images"
        if svm_size > 0:
            return seperateSVM(pData, pLabel, svm_size)
        else:
            return pLabel[:end_index], pData[:end_index]

    #dir3 = "pickled_images"+FILE_SEPARATOR+"ISH-noLearn_10000_15000_320_160.pkl.gz"#300_140.pkl.gz"
    dir3 = "pickled_images"+FILE_SEPARATOR+"ISH-noLearn_"+str(steps[1])+"_"+str(steps[2])+"_"+str(image_width)+"_"+str(image_height)+".pkl.gz"

    f3 = gzip.open(dir3, 'rb')
    train_set3, valid_set3, test_set3 = cPickle.load(f3)
    f3.close()
    print "after reading part 3"

    if end_index <= steps[2]:
        pData = np.concatenate((train_set1[0], train_set2[0], train_set3[0]), axis=0)
        print "Done Reading images"
        if svm_size > 0:
            return seperateSVM(pData, pLabel, svm_size)
        else:
            return pLabel[:end_index], pData[:end_index]

    # dir4 = "pickled_images" + FILE_SEPARATOR + "ISH-noLearn_15000_16352_320_160.pkl.gz"  # 300_140.pkl.gz"
    dir4 = "pickled_images"+FILE_SEPARATOR+"ISH-noLearn_"+str(steps[2])+"_"+str(steps[3])+"_"+str(image_width)+"_"+str(image_height)+".pkl.gz"

    f4 = gzip.open(dir4, 'rb')
    train_set4, valid_set4, test_set4 = cPickle.load(f4)
    f4.close()
    print "after reading part 4"

    if end_index <= steps[3] or steps[3] >= 16352:
        pData = np.concatenate((train_set1[0], train_set2[0], train_set3[0], train_set4[0]), axis=0)
        print "Done Reading images"
        if svm_size > 0:
            return seperateSVM(pData, pLabel, svm_size)
        else:
            return pLabel[:end_index], pData[:end_index]

    dir5 = "pickled_images" + FILE_SEPARATOR + "ISH-noLearn_" + str(steps[3]) + "_" + str(
        steps[4]) + "_" + str(image_width)+"_"+str(image_height)+".pkl.gz"

    f5 = gzip.open(dir5, 'rb')
    train_set5, valid_set5, test_set5 = cPickle.load(f5)
    f5.close()
    print "after reading part 5"

    pData = np.concatenate((train_set1[0], train_set2[0], train_set3[0], train_set4[0], train_set5[0]), axis=0)
    print "Done Reading images"
    if svm_size > 0:
        return seperateSVM(pData, pLabel, svm_size)
    else:
        return pLabel[:end_index], pData[:end_index]


#     # Generate new positive examples with noise
#     print "Add Positive examples multiple- ", MULTI
# #     count=0
#     countPositives = np.zeros(pLabel.shape[1])
#     positiveDataExamples = []
#     positivelabelExamples = []
#     negativeIndexes = []
#
#     countPositives_test = np.zeros(pLabel.shape[1])
#     positiveDataExamples_test = []
#     positivelabelExamples_test = []
#     for i in range(0, pData.shape[0]):
#         if any(pLabel[i, :]):
#             countPositives += pLabel[i, :]
#
#     for i in range(0, pData.shape[0]):
#         if any(pLabel[i, :]):
#             countPositives_test += pLabel[i, :]
#             useForTest = False
#             for j in range(0, countPositives_test.shape[0]):
#                 if pLabel[i, j] == 1 and (countPositives_test[j] > TRAIN_SPLIT*countPositives[j]):
#                     useForTest = True
#                     break
#
#             if toSplitPositives and useForTest:
#                 generate_positives(pData[i], pLabel[i], positiveDataExamples_test, positivelabelExamples_test, dropout_percent, MULTI, end_index)
#             else:
#                 generate_positives(pData[i], pLabel[i], positiveDataExamples, positivelabelExamples, dropout_percent, MULTI, end_index)
#         else:
#             negativeIndexes.append(i)
#
#     p = np.random.permutation(len(positiveDataExamples))
#     positiveDataExamples = np.array(positiveDataExamples)[p]
#     positivelabelExamples = np.array(positivelabelExamples)[p]
#     p = np.random.permutation(len(positiveDataExamples_test))
#     positiveDataExamples_test = np.array(positiveDataExamples_test)[p]
#     positivelabelExamples_test = np.array(positivelabelExamples_test)[p]
#     pData = pData[negativeIndexes]
#     print "add ", positivelabelExamples.shape[0]+positivelabelExamples_test.shape[0], "positive examples"
# #     print " to train set" if toSplitPositives else " to total set"
#
#     if not toSplitPositives:
#         split = np.floor(positiveDataExamples.shape[0]*TRAIN_SPLIT)
#         pData = np.concatenate((positiveDataExamples[:split],pData[:end_index],positiveDataExamples[split:]),axis=0)
#         pLabel = np.concatenate((positivelabelExamples[:split],pLabel[:end_index],positivelabelExamples[split:]),axis=0)
#     else:
#         #guarantee that positiveDataExamples_test will be in test set
#         end_index = np.max((end_index, (TRAIN_SPLIT*(positiveDataExamples.shape[0] + positiveDataExamples_test.shape[0]) - positiveDataExamples.shape[0])))
#
#         pData = np.concatenate((positiveDataExamples,pData[:end_index],positiveDataExamples_test),axis=0)
#         pLabel = np.concatenate((positivelabelExamples,pLabel[:end_index],positivelabelExamples_test),axis=0)
#
#
#     return pLabel, pData


def seperateSVM(pData, pLabel, svm_size):
    posRows = (pLabel != 0).sum(1) > 0
    posData = pData[posRows, :]
    posLabel = pLabel[posRows, :]
    print("Positive svm samples- ", posData.shape[0])
    negData = pData[~posRows[:svm_size + 200], :]
    negLabel = pLabel[~posRows[:svm_size + 200], :]
    print("Negative svm samples- ", negData.shape[0])
    # svm_data = np.concatenate((posData, negData[:svm_size-posData.shape[0]]), axis=0)
    # svm_label = np.concatenate((posLabel, negLabel[:svm_size-posData.shape[0]]), axis=0)
    svm_data = np.concatenate((posData, negData), axis=0)
    svm_label = np.concatenate((posLabel, negLabel), axis=0)
    return pLabel, pData, svm_data, svm_label


def generate_positives(positiveImage, labels, positiveDataArray, positivelabelsArray, dropout_percent=0.1, MULTI=30, end_index=0):
    for j in range(0, MULTI-1):
        dropout = np.random.binomial(1, 1-dropout_percent, size=positiveImage.shape[0])
        positiveDataArray.append(positiveImage*dropout)
        positivelabelsArray.append(labels)
    if end_index == 0 or MULTI == 0:
        positiveDataArray.append(positiveImage)
        positivelabelsArray.append(labels)


if __name__ == '__main__':
    pLabel, pData, svm_data, svm_label = pickleAllImages(svm_size=16351, num_labels=15, end_index=0, dropout_percent=0.3, MULTI=60)
    file_name = "piclked_articleCat_16351"
    try:
        cPickle.dump((svm_data, svm_label), open("pickled_temp/" + file_name + "-SVM.pkl.gz", 'wb'))
    except:
        pass
