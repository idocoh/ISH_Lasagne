function writeArticleBrainCatVectors(ishImage,imageId,cId,catName)
          
    global trg_dir go_genes_mat
%     hi =  go_genes_mat(cId,:);
    fulleGenesMat = full(go_genes_mat);
    cat = fulleGenesMat(cId',imageId);
    outPath  = strcat(trg_dir,catName,'\',ishImage.getFileName(),'_',catName,'.txt');
    dlmwrite(outPath,cat','delimiter',',')%'\n')
    
end