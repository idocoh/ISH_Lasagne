function [  ] = ReadImage(  )
%READIMAGE Summary of this function goes here
%   Detailed explanation goes here
Gene_Id = 10001;
GO = geneont('live',true); 
GO(Gene_Id).terms
ancestors = getancestors(GO,Gene_Id)
descendants = getdescendants(GO,Gene_Id)
relatives = getrelatives(GO,Gene_Id)
[matrix,id,relationship] = getmatrix(GO);
riboanc = GO(ancestors)

subontology = GO(relatives);
[cm acc rels] = getmatrix(subontology);
BG = biograph(cm, get(subontology.terms, 'name'));
% BG.nodes(acc==31966).Color = [1 0 0];
view(BG)
end

