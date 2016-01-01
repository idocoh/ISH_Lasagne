function writeArticleBrainCatVectors(ishImage,imageId,cId,catName)
          
    global trg_dir go_genes_mat
%     hi =  go_genes_mat(cId,:);
    cat = nnz(go_genes_mat(cId,imageId));
    outPath  = strcat(trg_dir,catName,'\',ishImage.getFileName(),'_',catName,'.txt');
    dlmwrite(outPath,cat,'delimiter',',')%'\n')
    
end