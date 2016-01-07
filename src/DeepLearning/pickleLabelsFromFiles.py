# http://scikit-learn.org/stable/tutorial/basic/tutorial.html
import cPickle as pickle
import numpy as np
from pickleImages import getTopCatVector

 
def readNewLables(lablesPath,categorysFileName,start_index=0,end_index=5000,TRAIN_DATA_PRECENT=0.8,VALIDATION_DATA_PRECENT=0.8):
    y = getTopCatVector(lablesPath+"\\*"+categorysFileName+".txt",start_index,end_index);  
    
    with open(categorysFileName+".pkl.gz",'wb') as f:
        pickle.dump(y, f, -1)
        f.close()
    
    dataAmount = end_index-start_index
    train_index = np.floor(dataAmount*TRAIN_DATA_PRECENT);
    validation_index = np.floor(dataAmount*VALIDATION_DATA_PRECENT)
    test_index = dataAmount
    # Divided dataset into 3 parts. 
    train_set_y = y[:train_index]
    val_set_y = y[train_index:validation_index]
    test_set_y = y[validation_index:]   
    
    return (train_set_y,test_set_y)   

def loadNewLabels(pcikledFilePath):
    with open(pcikledFilePath) as f:
        y = pickle.load(f)
        f.close()
    return y  
   
   
    
if __name__ == "__main__":
    labelDir = "C:\\Users\\Ido\\workspace\\ISH_Lasagne\\Doc\\all164cat"
    categorysFileName = "all164cat"
    y = readNewLables(labelDir,categorysFileName,end_index=16351)
    
#     labelDir = "C:\\Users\\Ido\\workspace\\ISH_Lasagne\\Doc\\all2081cat"
#     categorysFileName = "all2081cat"
#     y = readNewLables(labelDir,categorysFileName,end_index=16351)

  
