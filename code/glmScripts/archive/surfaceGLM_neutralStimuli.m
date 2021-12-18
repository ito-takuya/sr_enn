function output = surfaceGLM_neutralStimuli(subj, gsr, nproc)
% This function runs a GLM on the Glasser 2016 parcels on a single subject
% This will only regress out noise parameters and HRF convolved unique neutral stimuli (i.e., stimulus localizers) 
%

% Input parameter:
%   subj = subject number as a string
    
    numTasks = 6;
    numTRs = numTasks*745;

    data = loadSurfaceData64k_Task(subj);
    data = data.task;
    numROIs = size(data,1);
    

    % Load only noise regressors and task regressors for data
    X = loadStimFiles_neutralStimuli(subj,gsr);
    taskRegs = X.regressors; % These include the two binary regressors

    parfor (regionNum=1:numROIs,nproc)
        ROITimeseries = data(regionNum,:);
        %disp(['Regressing out region number ' num2str(regionNum) ' out of ' num2str(numROIs)])
        stats = regstats(ROITimeseries', taskRegs, 'linear', {'r', 'beta', 'rsquare', 'tstat'});

        % Collect regression results
        residual_dtseries(regionNum, :) = stats.r';
        betas_dtseries(regionNum, :) = stats.beta;
        t_dtseries(regionNum,:) = stats.tstat.t;
        p_dtseries(regionNum,:) = stats.tstat.pval;

    end

    % Write out 2D Residual dtseries to CSV
    if gsr==0
        outname1 = ['/projects3/StroopActFlow/data/results/glm_neutral_stimuli/' subj '_neutralStimuli_nuisanceResids_Surface64k.csv'];
        
    elseif gsr==1
        outname1 = ['/projects3/StroopActFlow/data/results/glm_neutral_stimuli/' subj '_neutralStimuli_nuisanceResids_Surface64k_noGSR.csv'];
    end
    outname2 = ['/projects3/StroopActFlow/data/results/glm_neutral_stimuli/' subj '_neutralStimuli_taskBetas_Surface64k.csv'];
    outname3 = ['/projects3/StroopActFlow/data/results/glm_neutral_stimuli/' subj '_neutralStimuli_taskTstat_Surface64k.csv'];
    outname4 = ['/projects3/StroopActFlow/data/results/glm_neutral_stimuli/' subj '_neutralStimuli_taskPval_Surface64k.csv'];

    csvwrite(outname1, residual_dtseries)
    csvwrite(outname2, betas_dtseries)
    csvwrite(outname3, t_dtseries)
    csvwrite(outname4, p_dtseries)
    output.residual_dtseries = residual_dtseries;
    output.betas_dtseries = betas_dtseries;
end



