function output = SurfaceGLM_64taskset(subj, gsr, nproc)
% This will only regress out noise parameters and HRF convolved 4 motor responses
%

% Input parameter:
%   subj = subject number as a string
    
    numTasks = 8;
    numTRs = numTasks*581;

    data = loadSurfaceData64k_Task(subj);
    data = data.task;
    numROIs = size(data,1);
    

    % Load only noise regressors and task regressors for data
    X = loadStimFiles_motorResp(subj,gsr);
    taskRegs = X.regressors; % These include the two binary regressors

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
        outname1 = ['/projects3/SRActFlow/data/results/GLM_MotorResponse/' subj '_motorResponse_nuisanceResids_Surface64k_noGSR.csv'];
        outname2 = ['/projects3/SRActFlow/data/results/GLM_MotorResponse/' subj '_motorResponse_taskBetas_Surface64k_noGSR.csv'];
    elseif gsr==1
        outname1 = ['/projects3/SRActFlow/data/results/GLM_MotorResponse/' subj '_motorResponse_nuisanceResids_Surface64k_GSR.csv'];
        outname2 = ['/projects3/SRActFlow/data/results/GLM_MotorResponse/' subj '_motorResponse_taskBetas_Surface64k_GSR.csv'];
    end

    csvwrite(outname1, residual_dtseries)
    csvwrite(outname2, betas_dtseries)
    output.residual_dtseries = residual_dtseries;
    output.betas_dtseries = betas_dtseries;
end



