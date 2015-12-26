function [ topCatIds , topCatNames ] = topBrainCatISHimages( numTopCat )
%USEISHIMAGES ISH images to be used
%   aspects                  1x2081            GO category aspect (F/C/P) - It's all P since we screened for GO process only
%   brain_cat_ids            1x164             GO category numeric ids for brain-related categories
%   brain_cat_names          164x1             GO category names for brain-related categories
%   cat_ids                  1x2081            GO category numeric ids
%   gene_names               16351x1           Gene names (symbols)
%   go_cat_names             2081x1            GO category names
%   go_genes_mat             2081x16351        Sparse matrix mapping genes to GO categories
%   mat_file_locations       16351x1           File names that match the images. Just replace the ending .mat with the ending .jpg
%
% The 16351 images listed in 'mat_file_locations' are the ones we used in our paper, the correspond to 15612 genes.

close all;

global full_go_genes cat_ids go_cat_names gene_names brain_cat_ids brain_cat_names
ISHimageClass.loadData();
%     load('onlyBrainISHrepresentation.mat')

% graphAmount(full_go_genes,' ',go_cat_names,gene_names)

[brain_gene_rows,~] = ismember(cat_ids',brain_cat_ids');

brain_go_genes = full_go_genes(brain_gene_rows,:);

% graphAmount(brain_go_genes,' Brain ',brain_cat_names,gene_names)
[topCatIndexs , topCatNames] = getTopCategories(numTopCat,brain_go_genes,brain_cat_names);
topCatIds = brain_cat_ids(topCatIndexs);
end

function [topCatIndexs , topCatNames] = getTopCategories(numTopCat,brain_go_genes,brain_cat_names)

numGensInCat = sum(brain_go_genes,2);
[numGensInCatSorted,SortedIndexs] = sort(numGensInCat,'descend');

topCatIndexs = SortedIndexs(1:numTopCat);
topCatNames = brain_cat_names(topCatIndexs);

end

