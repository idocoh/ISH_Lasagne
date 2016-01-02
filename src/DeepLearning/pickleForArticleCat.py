import cPickle
import gzip
import numpy

def pickleAllImages(outputDir):
    dir1="C:\Users\Ido\workspace\ISH_Lasagne\src\DeepLearning\pickled_images\ISH-noLearn_0_5000_300_140.pkl.gz"
    f1 = gzip.open(dir1, 'rb')
    train_set1, valid_set1, test_set1 = cPickle.load(f1)
    f1.close()
    print "after reading dir1"
    dir2="C:\Users\Ido\workspace\ISH_Lasagne\src\DeepLearning\pickled_images\ISH-noLearn_5001_11000_300_140.pkl.gz"
    f2 = gzip.open(dir2, 'rb')
    train_set2, valid_set2, test_set2 = cPickle.load(f2)
    f2.close()
    print "after reading dir2"

#     dir3="C:\Users\Ido\workspace\ISH_Lasagne\src\DeepLearning\pickled_images\ISH-noLearn_11001_16351_300_140.pkl.gz"
#     f3 = gzip.open(dir3, 'rb')
#     train_set3, valid_set3, test_set3 = cPickle.load(f3)
#     f3.close()   
    
#     ,train_set2[0],test_set2[0]
    train_set2 = numpy.concatenate((train_set1[0],test_set1[0],train_set2[0],test_set2[0]),axis=0) 
    
#     del train_set1, valid_set1, test_set1, train_set1, valid_set1, test_set1
#     data = numpy.concatenate((train_set1[0],test_set1[0],train_set2[0],test_set2[0],train_set3[0],test_set3[0]),axis=0) 
    print "writing to file"
    f = gzip.open(outputDir,'wb')
    cPickle.dump(train_set2[:6000], f, protocol=2)
    f.close()
#     # Divided dataset into 3 parts. 
#     dataAmount = end_index-start_index
#     train_index = numpy.floor(dataAmount*TRAIN_DATA_PRECENT);
#     validation_index = numpy.floor(dataAmount*VALIDATION_DATA_PRECENT)
#     test_index = dataAmount
# 
#     train_set_x = pData[:train_index]
#     val_set_x = pData[train_index:validation_index]
#     test_set_x = pData[validation_index:]
#     train_set_y = pLabel[:train_index]
#     val_set_y = pLabel[train_index:validation_index]
#     test_set_y = pLabel[validation_index:]
#     
#     train_set = train_set_x, train_set_y
#     val_set = val_set_x, val_set_y
#     test_set = test_set_x, test_set_y    

if __name__ == '__main__':
    
    dir = "C:\\Users\\Ido\\workspace\\ISH_Lasagne\\src\\DeepLearning\\pickled_images\\0_11000_all.pkl.gz"
    pickleAllImages(dir)


