function [ SUCCESS ] = copyBrainImages(ishImage)

    global src_dir trg_dir
       
    src_file = strcat(src_dir,ishImage.file_location{1});
    [SUCCESS,MESSAGE,MESSAGEID] = copyfile(src_file,trg_dir);
       
    if (~SUCCESS)
        disp(MESSAGE);
    end
end