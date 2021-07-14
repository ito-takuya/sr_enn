function output = surfaceGLM_rest(subj, gsr, nproc)
% This function runs a GLM on the entire surface (i.e., all vertices, not just the parcellated vertex) on a single subject
% This will only regress out noise parameters. 
% Only runs on task data (all 8 tasks, and concatenates them). Also demeans each run prior to concatenation
%
% Original purpose: To run informational connectivity analysis on Auditory versus Visual task rules
%
% Input parameter:
%   subj = subject number as a string
   
    addpath('/projects/AnalysisTools/gifti-1.6/')

    numTRs = 1070;
    basedir = ['/projects3/StroopActFlow/data/' subj '/'];

    %%

    datadir = [basedir 'analysis/'];
    datafile = [datadir 'Rest1_Atlas_64k.dtseries.nii'];
    %disp(['Loading in run ' num2str(task) ' out of 8'])
    data = ciftiopen(datafile,'wb_command');
    data = data.cdata;

    %%
    
    numVertices = size(data,1);
    
    % Load only noise regressors for the rest data
    X = loadStimFiles_rest(subj,gsr);
    taskRegs = X.noiseRegressors; 

    parfor (regionNum=1:numVertices, nproc)
        ROITimeseries = data(regionNum,:);
        %disp(['Regressing out region number ' num2str(regionNum) ' out of ' num2str(numVertices)])
        stats = regstats(ROITimeseries', taskRegs, 'linear', {'r', 'beta', 'rsquare'});

        % Collect regression results
        residual_dtseries(regionNum, :) = stats.r';
        betas_dtseries(regionNum, :) = stats.beta;

    end

    % Write out 2D Residual dtseries to CSV
    if gsr==0
        outname1 = ['/projects3/StroopActFlow/data/results/rest_glm_64k/' subj '_rest_nuisanceResids_64kSurface.csv'];
        
    elseif gsr==1
        outname1 = ['/projects3/StroopActFlow/data/results/rest_glm_64k/' subj '_rest_nuisanceResids_64kSurface_GSR.csv'];
    end

    outname2 = ['/projects3/StroopActFlow/data/results/rest_glm_64k/' subj '_rest_taskbetas_64kSurface.csv'];

    csvwrite(outname1, residual_dtseries)
    csvwrite(outname2, betas_dtseries)
    output.residual_dtseries = residual_dtseries;
    output.betas_dtseries = betas_dtseries;
end



