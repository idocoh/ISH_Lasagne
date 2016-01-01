function writeEachBrainCatVectors(ishImage,catNames)
          
    global trg_dir
    allCat = ishImage.getTopCatVector();
    for i=1:164
        catName = catNames{i};
        cat = allCat(i);
        outPath  = strcat(trg_dir,catName,'\',ishImage.getFileName(),'_',catName,'.txt');
        dlmwrite(outPath,cat,'delimiter',',')%'\n')
    end
end