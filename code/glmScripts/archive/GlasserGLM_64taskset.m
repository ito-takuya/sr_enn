function output = GlasserGLM_64taskset(subj, gsr, nproc)
% This function runs a GLM on the Glasser 2016 parcels on a single subject
% This will only regress out noise parameters and HRF convolved unique 64 task set 
%

% Input parameter:
%   subj = subject number as a string
    
    numTasks = 8;
    numTRs = numTasks*581;

    data = loadGlasserData(subj);
    data = data.task;
    numROIs = size(data,1);
    

    % Load only noise regressors and task regressors for data
    X = loadStimFiles_by64TaskSet_mc16(subj,gsr);
    taskRegs = X.taskRegressors; % These include the two binary regressors

    parfor (regionNum=1:numROIs,nproc)
        ROITimeseries = data(regionNum,:);
        %disp(['Regressing out region number ' num2str(regionNum) ' out of ' num2str(numROIs)])
        stats = regstats(ROITimeseries', taskRegs, 'linear', {'r', 'beta', 'rsquare'});

        % Collect regression results
        residual_dtseries(regionNum, :) = stats.r';
        betas_dtseries(regionNum, :) = stats.beta;

    end

    % Write out 2D Residual dtseries to CSV
    if gsr==0
        outname1 = ['/projects2/ModalityControl2/data/resultsGlasser/glm64Taskset/' subj '_64taskset_nuisanceResids_Glasser.csv'];
        outname2 = ['/projects2/ModalityControl2/data/resultsGlasser/glm64Taskset/' subj '_64taskset_taskBetas_Glasser.csv'];
        
    elseif gsr==1
        outname1 = ['/projects2/ModalityControl2/data/resultsGlasser/glm64Taskset/' subj '_64taskset_nuisanceResids_Glasser_GSR.csv'];
        outname2 = ['/projects2/ModalityControl2/data/resultsGlasser/glm64Taskset/' subj '_64taskset_taskBetas_Glasser_GSR.csv'];
    end

    csvwrite(outname1, residual_dtseries)
    csvwrite(outname2, betas_dtseries)
    output.residual_dtseries = residual_dtseries;
    output.betas_dtseries = betas_dtseries;
end



