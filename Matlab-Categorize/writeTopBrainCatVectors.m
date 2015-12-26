function writeTopBrainCatVectors(ishImage)
          
    global trg_dir
    outPath  = strcat(trg_dir,ishImage.getFileName(),'_TopCat','.txt');
    dlmwrite(outPath,ishImage.getTopCatVector(),'delimiter',',')%'\n')

end