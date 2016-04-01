import cPickle
import gzip
import numpy as np
import platform



def pickleAllImages(num_labels,TRAIN_SPLIT=0.8,end_index=16351,dropout_percent=0.1,MULTI=20, toSplitPositives = True):
#     if platform.dist()[0]:
# #         FILE_SEPARATOR = "\\"
#     else :
    FILE_SEPARATOR="/"

    if num_labels==15:
        labelsFile = "pickled_images"+FILE_SEPARATOR+"articleCat.pkl.gz"
    elif num_labels==20:
        labelsFile = "pickled_images"+FILE_SEPARATOR+"topCat.pkl.gz"
    elif num_labels==164:
        labelsFile = "pickled_images"+FILE_SEPARATOR+"all164cat.pkl.gz"
    elif num_labels==2081:
        labelsFile = "pickled_images"+FILE_SEPARATOR+"all2081cat.pkl.gz"
    else:
        print "bad label path!!!!!!!"
         
    print "Loading labels..."
    with open(labelsFile) as l:
        pLabel = cPickle.load(l)
        l.close()
    
    dir1 = "pickled_images"+FILE_SEPARATOR+"ISH-noLearn_0_5000_300_140.pkl.gz"
    f1 = gzip.open(dir1, 'rb')
    train_set1, valid_set1, test_set1 = cPickle.load(f1)
    f1.close()
    print "after reading part 1"
    
#     if end_index <= 5000:
#         pData =  np.concatenate((train_set1[0],test_set1[0]),axis=0) 
 
#     else :
    dir2 = "pickled_images"+FILE_SEPARATOR+"ISH-noLearn_5000_11000_300_140.pkl.gz"
    f2 = gzip.open(dir2, 'rb')
    train_set2, valid_set2, test_set2 = cPickle.load(f2)
    f2.close()
    print "after reading part 2"
            
#         if end_index <= 11000:
#             pData = np.concatenate((train_set1[0],test_set1[0],train_set2[0],test_set2[0]),axis=0) 
#         else:
    dir3 = "pickled_images"+FILE_SEPARATOR+"ISH-noLearn_11000_16352_300_140.pkl.gz"
    f3 = gzip.open(dir3, 'rb')
    train_set3, valid_set3, test_set3 = cPickle.load(f3)
    f3.close()   
    print "after reading part 3"
           
    pData = np.concatenate((train_set1[0], test_set1[0], train_set2[0], test_set2[0], train_set3[0], test_set3[0]), axis = 0)
    
    return pLabel[:end_index], pData[:end_index]
#     f = gzip.open("images_16351_300_140",'wb')
#     cPickle.dump(pData, f, protocol=2)
#     f.close()
    
    print "Add Positive examples multiple- ", MULTI
#     count=0
    countPositives = np.zeros(pLabel.shape[1])
    positiveDataExamples = []
    positivelabelExamples = []
    negativeIndexes = []
    
    countPositives_test = np.zeros(pLabel.shape[1])    
    positiveDataExamples_test = []
    positivelabelExamples_test = []
    for i in range(0,pData.shape[0]):
        if any(pLabel[i,:]):
            countPositives += pLabel[i,:]
            
            
    for i in range(0,pData.shape[0]):
        if any(pLabel[i,:]):   
            countPositives_test += pLabel[i,:]
            useForTest = False
            for j in range(0,countPositives_test.shape[0]):
                if pLabel[i,j] == 1 and (countPositives_test[j] > TRAIN_SPLIT*countPositives[j]):
                    useForTest = True
                    break
            
            if toSplitPositives and useForTest:
                generatePositives(pData[i],pLabel[i],positiveDataExamples_test,positivelabelExamples_test,dropout_percent,MULTI,end_index) 
            else:
                generatePositives(pData[i],pLabel[i],positiveDataExamples,positivelabelExamples,dropout_percent,MULTI,end_index) 
        else:
            negativeIndexes.append(i)
            
    p = np.random.permutation(len(positiveDataExamples))
    positiveDataExamples = np.array(positiveDataExamples)[p]
    positivelabelExamples = np.array(positivelabelExamples)[p]
    p = np.random.permutation(len(positiveDataExamples_test))
    positiveDataExamples_test = np.array(positiveDataExamples_test)[p]
    positivelabelExamples_test = np.array(positivelabelExamples_test)[p]
    pData = pData[negativeIndexes]
#     print count
#     print countPositives
#     print positiveDataExamples
    print "add ", positivelabelExamples.shape[0]+positivelabelExamples_test.shape[0], "positive examples" 
#     print " to train set" if toSplitPositives else " to total set"

    if not toSplitPositives:
        split = np.floor(positiveDataExamples.shape[0]*TRAIN_SPLIT)
        pData = np.concatenate((positiveDataExamples[:split],pData[:end_index],positiveDataExamples[split:]),axis=0)
        pLabel = np.concatenate((positivelabelExamples[:split],pLabel[:end_index],positivelabelExamples[split:]),axis=0)
    else:
        #guarantee that positiveDataExamples_test will be in test set
        end_index = np.max((end_index, (TRAIN_SPLIT*(positiveDataExamples.shape[0] + positiveDataExamples_test.shape[0]) - positiveDataExamples.shape[0])))

        pData = np.concatenate((positiveDataExamples,pData[:end_index],positiveDataExamples_test),axis=0)
        pLabel = np.concatenate((positivelabelExamples,pLabel[:end_index],positivelabelExamples_test),axis=0)

    
    return pLabel, pData 

def generatePositives(positiveImage, labels, positiveDataArray, positivelabelsArray,dropout_percent=0.1,MULTI=30,end_index=0):
    for j in range(0,MULTI-1):
        dropout = np.random.binomial(1, 1-dropout_percent, size=positiveImage.shape[0])
        positiveDataArray.append(positiveImage*dropout)
        positivelabelsArray.append(labels)
    if end_index==0 or MULTI==0:
        positiveDataArray.append(positiveImage)
        positivelabelsArray.append(labels)
        
#      np.random.choice([0, 1], size=(layer_1.shape[0],), p=[dropout_percent, 1-dropout_percent])
#     np.random.shuffle(arr)

if __name__ == '__main__':
    pickleAllImages(num_labels=15,end_index=0,dropout_percent=0.3,MULTI=60)

