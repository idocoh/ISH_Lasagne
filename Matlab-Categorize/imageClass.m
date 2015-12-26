function [ output_args ] = imageClass( mat_file_location_id )
%IMAGECLASS Summary of this function goes here
%   aspects                  1x2081            GO category aspect (F/C/P) - It's all P since we screened for GO process only
%   brain_cat_ids            1x164             GO category numeric ids for brain-related categories
%   brain_cat_names          164x1             GO category names for brain-related categories
%   cat_ids                  1x2081            GO category numeric ids
%   gene_names               16351x1           Gene names (symbols)
%   go_cat_names             2081x1            GO category names
%   go_genes_mat             2081x16351        Sparse matrix mapping genes to GO categories
%   mat_file_locations       16351x1           File names that match the images. Just replace the ending .mat with the ending .jpg

mat_file_location;
gene_name;
go_cat_ids;
go_cat_names;



end

