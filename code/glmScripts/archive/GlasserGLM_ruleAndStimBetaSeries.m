function output = GlasserGLM_ruleAndStimBetaSeries(subj, gsr, nproc)
% This function runs a GLM on the Glasser 2016 parcels on a single subject
% This will only regress out noise parameters and HRF convolved unique neutral stimuli (i.e., stimulus localizers) 
%

% Input parameter:
%   subj = subject number as a string
    
    numTasks = 6;
    numTRs = numTasks*745;

    data = loadGlasserData(subj);
    data = data.task;
    numROIs = size(data,1);
    

    % Load only noise regressors and task regressors for data
    X = loadStimFiles_RuleAndStimBetaSeries(subj,gsr);
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
        outname1 = ['/projects3/StroopActFlow/data/results/GlasserResults/glm_ruleStimBetaSeries/' subj '_RuleAndStimBetaSeries_nuisanceResids_Surface64k.csv'];
        
    elseif gsr==1
        outname1 = ['/projects3/StroopActFlow/data/results/GlasserResults/glm_ruleStimBetaSeries/' subj '_RuleAndStimBetaSeries_nuisanceResids_Surface64k_noGSR.csv'];
    end
    outname2 = ['/projects3/StroopActFlow/data/results/GlasserResults/glm_ruleStimBetaSeries/' subj '_RuleAndStimBetaSeries_taskBetas_Surface64k.csv'];

    csvwrite(outname1, residual_dtseries)
    csvwrite(outname2, betas_dtseries)
    output.residual_dtseries = residual_dtseries;
    output.betas_dtseries = betas_dtseries;
end



