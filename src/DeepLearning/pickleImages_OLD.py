import os
from PIL import Image
from numpy import genfromtxt
import gzip, cPickle
from glob import glob
import numpy as np
import pandas as pd
from random import randint
import time
from logistic_sgd import load_data
from nolearnNN import run_All

NUM_IMAGES = 50
IMAGE_WIDTH = 300
IMAGE_HEIGHT = 140

class pickleImages(object):   
    '''
    classdocs
    '''


    def __init__(self, dirName):
        '''
        Constructor
        '''
        self.dir = dirName 


    def getImageNames(self,dirName):
        imagesArray = []
        count=0
        for root, subdirs, files in os.walk(dir):
            for file in files:
                if os.path.splitext(file)[1].lower() in ('.jpg', '.jpeg'):
#                      imagesArray.append(file)
                    if count<NUM_IMAGES :
                         imagesArray.append(randint(0,9))
                         count+=1
    #                  print os.path.join(root, file)                 
        return np.array(imagesArray)

#     def picleImages(imagesArray):
#         data = []
#         for imageName in imagesArray:
#             im = Image.open(imageName)
#             data.append([im,imageName[0:-4]])
#             
#         print data      
          
    
    def pickleData(self,start_index,end_index):
        Data = dir_to_dataset(dir+"\\*.jpg",start_index,end_index)
#         Data = fromPickledData("ISH.pkl.gz")
#         Data =[]
        
        #TODO: fix y!!!! 
        y = self.getImageNames(dir);
        
        dataAmount = end_index-start_index
        train_index = np.floor(dataAmount*0.8);
        validation_index = np.floor(dataAmount*0.9)
        test_index = dataAmount
#         # Divided dataset into 3 parts. 
        train_set_x = Data[:train_index]
        val_set_x = Data[train_index:validation_index]
        test_set_x = Data[validation_index:]
        train_set_y = y[:train_index]
        val_set_y = y[train_index:validation_index]
        test_set_y = y[validation_index:]
        
        train_set = train_set_x, train_set_y
        val_set = val_set_x, val_set_y
        test_set = test_set_x, test_set_y
        
        dataset = [train_set, val_set, test_set]
        
#         f = gzip.open('ISH-noLearn_all_300_140.pkl.gz','wb')
        f = gzip.open('pickled_images/ISH-noLearn_'+str(start_index)+'_'+str(end_index)+'_'+str(IMAGE_WIDTH)+'_'+str(IMAGE_HEIGHT)+'.pkl.gz','wb')
        cPickle.dump(dataset, f, protocol=2)
        f.close()     
        
def fromPickledData(zipName):
    datasets = load_data(zipName)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    Data = train_set_x + valid_set_x + test_set_x
#     Data = []
#     for i in xrange(train_set_x.get_value(borrow=True).shape[0]):
#         Data.append(train_set_x[i])
#     for i in xrange(valid_set_x.get_value(borrow=True).shape[0]):
#         Data.append(valid_set_x[i])
#     for i in xrange(test_set_x.get_value(borrow=True).shape[0]):
#         Data.append(test_set_x[i])
    
    return Data

def dir_to_dataset(glob_files,start_index,end_index,loc_train_labels=""):
        print("Gonna process:\n\t %s"%glob_files)
        dataset = []
        for file_count, file_name in enumerate( sorted(glob(glob_files),key=len) ):
            if file_count < start_index or end_index-1 < file_count :
                continue;
            image = Image.open(file_name).convert('L') #tograyscale
#             print image.format,image.size, image.mode
#             image.show()
            #resize
            
            wpercent = (IMAGE_WIDTH/float(image.size[0]))
            hsize = int((float(image.size[1])*float(wpercent)))
            image = image.resize((IMAGE_WIDTH,IMAGE_HEIGHT), Image.ANTIALIAS)
#             image.thumbnail((),Image.ANTIALIAS)
#             print image.format,image.size, image.mode
#             img = Image.open(file_name).convert('LA') #tograyscale
#             pixels = [f[0] for f in list(image.getdata())] #f[0]
#             image.show()

            #convert to float
            pixels = list(image.getdata())
            float_pixels  = [float(x) / 255 for x in pixels]
            dataset.append(float_pixels)
#             print float_pixels
        
            if file_count % 10 == 0:
                print("\t %s files processed"%file_count)
                
#             if file_count > NUM_IMAGES:
#                 break
            
        # outfile = glob_files+"out"
        # np.save(outfile, dataset)
        if len(loc_train_labels) > 0:
            df = pd.read_csv(loc_train_labels)
            return np.array(dataset), np.array(df["Class"])
        else:
            return np.array(dataset)
    
    
if __name__ == '__main__':
    start_time=time.clock()
    dir = "C:\\Users\\Ido\\Pictures\\ISH-images-mouse"
    pick = pickleImages(dir)
    pick.pickleData(0,NUM_IMAGES)
    print "time"
    print time.clock()-start_time
#     
#     run_All()

#     test_SdA()