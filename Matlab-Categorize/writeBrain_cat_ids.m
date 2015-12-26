function writeBrain_cat_ids(ishImage)
          
    global trg_dir
    outPath  = strcat(trg_dir,ishImage.getFileName(),'.txt');
    dlmwrite(outPath,ishImage.gene_brain_cat_ids,'-append','delimiter','\n')

end