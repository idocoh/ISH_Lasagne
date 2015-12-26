import cPickle as pickle

def writeDataToFile(pcikledFilePath, FILE_NAME,labelNumber=0):
    with open(pcikledFilePath) as f:
        ob = pickle.load(f)
        f.close()
    params , labels = ob;
    
    
#     FILE_NAME = "results/"+outputFileName
    outputFile = open(FILE_NAME, "w");
   
    for exapmle in zip(params,labels):
        p = exapmle[0];
        l = exapmle[1];
        
        if (labelNumber==0) :
            labelString = ','.join(map(str, map(int, l)))
        elif (labelNumber <= l.size):
            labelString = str(int(l[labelNumber-1]))
        else:
            labelString="?"
            
        outputFile.write(labelString)
        
        index=1;
        for x in p:
            outputFile.write(" " + str(index) + ":" + str(x))
            index+=1
        
        outputFile.write("\n")
          

    
if __name__ == "__main__":
    pickName = "C:\\Users\\Ido\\workspace\\ISH_Lasagne\\src\\DeepLearning\\results\\noLearn_50_3_hiddenLayerOutput_0.pickle"
    writeDataToFile(pickName,"myData.txt")

    