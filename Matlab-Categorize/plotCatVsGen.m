function [ mu1 mu2 ] = plotCatVsGen( )
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

graphAmount(full_go_genes,' ',go_cat_names,gene_names)

[brain_gene_rows,~] = ismember(cat_ids',brain_cat_ids');

brain_go_genes = full_go_genes(brain_gene_rows,:);

graphAmount(brain_go_genes,' Brain ',brain_cat_names,gene_names)

end

function graphAmount(go_genes_matrix,prefix_title,cat_vector,gene_vector)

f = figure()
subplot(2,1,1)
graphWithMean(2,cat_vector,'Category index','Amount',strcat('Number of Genes In Each ',prefix_title,' Category'));

subplot(2,1,2)
graphWithMean(1,gene_vector,'Gene index','Amount',strcat('Number of ',prefix_title,' Categorys For Each Gene'));

src_dir = 'C:\Users\Abigail\Desktop\Ido\';
file_name = strcat(src_dir,prefix_title,' Catagorys&Gene Amount','.jpg');
% saveas(f,file_name);
    function graphWithMean(dim,x_vector,xLabel,yLabel,tle)
                
        amount = sum(go_genes_matrix,dim);
        mu = mean(amount);
        
        bar(1:length(x_vector),amount);
        hline = refline([0 mu]);
        set(hline,'Color','r');
        text(length(x_vector)+ 10,mu + 15,strcat('Mean- ',num2str(mu)),'LineWidth',4,'Color','r');
        xlabel(xLabel);
        ylabel(yLabel);
        title(tle);
        % Labels = x;
        % set(gca, 'XTick', 1:length(numCategoryForEachGene), 'XTickLabel', Labels);
        
    end
end
