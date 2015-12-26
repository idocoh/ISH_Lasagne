classdef ISHimageClass < handle
    %UNTITLED Summary of this class goes here
    %   aspects                  1x2081            GO category aspect (F/C/P) - It's all P since we screened for GO process only
    %   brain_cat_ids            1x164             GO category numeric ids for brain-related categories
    %   brain_cat_names          164x1             GO category names for brain-related categories
    %   cat_ids                  1x2081            GO category numeric ids
    %   gene_names               16351x1           Gene names (symbols)
    %   go_cat_names             2081x1            GO category names
    %   go_genes_mat             2081x16351        Sparse matrix mapping genes to GO categories
    %   mat_file_locations       16351x1           File names that match the images. Just replace the ending .mat with the ending .jpg
    
    
    properties
        file_id;
        file_location;
        gene_name;
        gene_go_cat_ids;
        gene_go_cat_names;
        gene_brain_cat_ids;
        gene_brain_cat_names;
        %         image;
    end
    
    
    
    methods (Access = private)
        
        function init(obj)
%             obj.loadData();
            obj.file_location = obj.calcFileLocation();
            obj.gene_name = obj.calcGeneName();
            [obj.gene_go_cat_ids , obj.gene_go_cat_names] = obj.calcGoIdsAndNames();
            [obj.gene_brain_cat_ids , obj.gene_brain_cat_names] = obj.calcBrainIdsAndNames();
            %             obj.image = readImage();
            
        end
        
        function file_location = calcFileLocation(obj)
            global mat_file_locations
            
            folder_path = '';
            mat_file_location = mat_file_locations(obj.file_id);
            file_location = strrep(mat_file_location, '.mat', '.jpg');
            file_location = strcat(folder_path , file_location);
            
        end
        
        function name = calcGeneName(obj)
            global gene_names
            
            name = gene_names(obj.file_id);
            
        end
        
        function [ids , names] = calcGoIdsAndNames(obj)
            global cat_ids go_cat_names full_go_genes
            
            go_indexs = full_go_genes(:,obj.file_id);
            ids = cat_ids(go_indexs);
            names = go_cat_names(go_indexs);
            
        end
        
        function [ids , names] = calcBrainIdsAndNames(obj)
            global brain_cat_ids brain_cat_names

            [is_go_brain,brain_indexs] = ismember(obj.gene_go_cat_ids,brain_cat_ids);
            brain_indexs = brain_indexs(brain_indexs~=0);
            ids = brain_cat_ids(brain_indexs);
            names = brain_cat_names(brain_indexs);
            
        end
        
        
        
    end
    
    methods
        
        function Eobj = ISHimageClass(file_id)
            Eobj.file_id = file_id;
            Eobj.init();
            
        end
        
        function image = readImage(obj)
            
            image = imread(obj.file_location{1});
            
        end
        
        function file_name = getFileName(obj)
            
            file_name = strrep(obj.file_location , '.jpg' , '');
            file_name = file_name{1};
        end
        
        function topCatVector = getTopCatVector(obj)
             global topCatIds
             topCatVector = ismember(topCatIds,obj.gene_brain_cat_ids);
             
%              if any(topCatVector)
%                  disp(topCatVector);
%              end
        end
        
    end
    
    methods (Static)
        function loadData()
            global brain_cat_ids brain_cat_names cat_ids gene_names 
            global go_cat_names go_genes_mat mat_file_locations full_go_genes
            
            filePath = 'images_go_genes_mat_screened.mat';
            load(filePath);
                     
%             brain_cat_ids
%             brain_cat_names
%             cat_ids
%             gene_names
%             go_cat_names
%             go_genes_mat
%             mat_file_locations
            
            full_go_genes = logical(full(go_genes_mat));

        end
    end
    
end

