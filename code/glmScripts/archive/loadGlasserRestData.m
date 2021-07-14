function output = loadGlasserRestData(subj)
% Taku Ito
% 3/27/18
%
% This script loads in the surface data using the Glasser et al. 2016 360 ROI surface parcellation scheme
% 
% Parameters: subj ( must be in put with single quotations, i.e., as a string)

    % Get data directory based on the assumption that script is in projects/IndivRITL/docs/scripts/modalControl/
    datadir = ['/projects3/SRActFlow/data/'];
    analysisdir = [datadir '/' subj '/analysis/'];

    % Each dtseries is 380x581, so we will want to create an empty matrix of 380x4648 (8*581 = 4648)
    % We will create a 3d Matrix first, and reshape it into a 2d roi x dtseries after
    nrois = 360;
    
    % Parcellate and import rest data too
    inFile = [datadir subj '/analysis/Rest1_Atlas.LR.Glasser2016Parcels.32k_fs_LR.ptseries.nii'];

    % Now import the data to MATLAB using ft_read_cifti in AnalysisTools
    disp(['Importing parcellated cifti file for rest data'])
    tmp = ciftiopen(inFile, 'wb_command');
    rest = tmp.cdata;

    % Reshape into two dimensions, nrois x 4648
    output.rest = rest;

end
