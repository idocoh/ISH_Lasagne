import os
from PIL import Image
from numpy import genfromtxt
import gzip, cPickle
from glob import glob
import numpy as np
import pandas as pd
from random import randint
from logistic_sgd import load_data
from numpy import loadtxt
import time
from nolearnNN import run_All

Image.MAX_IMAGE_PIXELS = None

NUM_IMAGES = 50
IMAGE_WIDTH = 300
IMAGE_HEIGHT = 140
TRAIN_DATA_PRECENT = 0.8
VALIDATION_DATA_PRECENT = TRAIN_DATA_PRECENT + 0.0


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
        for root, subdirs, files in os.walk(dir):
            for file in files:
                if os.path.splitext(file)[1].lower() in ('.jpg', '.jpeg'):
#                      imagesArray.append(file)
                     imagesArray.append(randint(0,9))
    #                  print os.path.join(root, file)                 
        return imagesArray
    
    

#     def picleImages(imagesArray):
#         data = []
#         for imageName in imagesArray:
#             im = Image.open(imageName)
#             data.append([im,imageName[0:-4]])
#             
#         print data      
          
    
    def pickleData(self,start_index,end_index):
      
        y=1
#         y = getTopCatVector(self.dir+"\\*TopCat.txt",start_index,end_index)
               
#         fy = gzip.open('BraionISH_TopCAT.pkl.gz','wb')
#         cPickle.dump(y, fy, protocol=2)
#         fy.close()   
          
#         fy = gzip.open('BraionISH_TopCAT.pkl.gz', 'rb')
#         y = cPickle.load(fy)
#         fy.close()
    
        Data = dir_to_dataset(self.dir+"\\*.jpg",start_index,end_index)
#         fd = gzip.open('BraionISH_Data.pkl.gz','wb')
#         cPickle.dump(Data, fd, protocol=2)
#         fd.close()

#         Data = fromPickledData("ISH-origin.pkl.gz")
 
        # Divided dataset into 3 parts. 
        dataAmount = end_index-start_index
        train_index = np.floor(dataAmount*TRAIN_DATA_PRECENT);
        validation_index = np.floor(dataAmount*VALIDATION_DATA_PRECENT)
        test_index = dataAmount
#         # Divided dataset into 3 parts. 
        train_set_x = Data[:train_index]
        val_set_x = Data[train_index:validation_index]
        test_set_x = Data[validation_index:]
        train_set_y = y#[:train_index]
        val_set_y = y#[train_index:validation_index]
        test_set_y = y#[validation_index:]
        
        train_set = train_set_x, train_set_y
        val_set = val_set_x, val_set_y
        test_set = test_set_x, test_set_y
        
        dataset = [train_set, val_set, test_set]
        
#         f = gzip.open('ISH-noLearn_all_300_140.pkl.gz','wb')
        pickledPath = 'pickled_images/ISH-noLearn_'+str(start_index)+'_'+str(end_index)+'_'+str(IMAGE_WIDTH)+'_'+str(IMAGE_HEIGHT)+'.pkl.gz'
        f = gzip.open(pickledPath,'wb')
        cPickle.dump(dataset, f, protocol=2)
        f.close()   
        return pickledPath
        
def fromPickledData(zipName):
    f = gzip.open(zipName, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    
    train_set_x, train_set_y = train_set
    valid_set_x, valid_set_y = valid_set
    test_set_x, test_set_y = test_set
        
#     datasets = load_data(zipName)
#     train_set_x, train_set_y = datasets[0]
#     valid_set_x, valid_set_y = datasets[1]
#     test_set_x, test_set_y = datasets[2]

    Data = np.concatenate((train_set_x, valid_set_x, test_set_x), axis=0) #train_set_x + valid_set_x + test_set_x

#     fd = gzip.open('BraionISH_TopCAT.pkl.gz', 'rb')
#     Data = cPickle.load(fd)
#     fd.close()
    
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

def getTopCatVector(glob_files,start_index,end_index):
        
        print("Gonna process category:\n\t %s"%glob_files)
        topCatVectorSet = []
        for file_count, file_name in enumerate( sorted(glob(glob_files),key=len) ):
            if file_count < start_index or end_index-1 < file_count :
                continue;        
#             f = open(file_name, 'r')
#             catVectocr = f.read().split(',') #f.readlines()
            catVectocr = loadtxt(file_name, comments="#", delimiter=",", unpack=False)
            float_vector  = [np.float32(x) for x in catVectocr]
            topCatVectorSet.append(float_vector)
#             print float_vector
#                      imagesArray.append(randint(0,9))
    #                  print os.path.join(root, file)                 
        return np.array(topCatVectorSet)
            
def runPickleImages(dir,startIndex=0,endIndex=NUM_IMAGES):
    start_time=time.clock()
    
    pick = pickleImages(dir)
    pickledPath = pick.pickleData(startIndex,endIndex)
    print "time"
    print time.clock()-start_time
    
    return pickledPath
    
    
if __name__ == '__main__':#     dir = "C:\\Users\\Abigail\\Desktop\\Ido\\pyWS\\First\\G_images\\"
#     dir = "C:\\Users\\Abigail\\Desktop\\Ido\\BrainISHimages"
#     pick = pickleImages(dir)
#     pick.pickleData()
    
    dir = "C:\Users\Ido\Pictures\BrainISHimages"
    runPickleImages(dir)
#     test_SdA()