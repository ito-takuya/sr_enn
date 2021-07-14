function output = loadStimFiles_neutralStimuli(subj, gsr)
% Taku Ito
% 01/13/2017
%
% This script imports the stim files for IndivRITL modality control into a regressor matrix including wm/ventricle/wholebrain timeseries, 12 motion parameters, and 4 stimulus time series
% Regressing out WM, Ventricles (and corresponding derivatives), motion parameters, linear trend, and 4 neutral stimuli (stimulus localizers)
% 
% Parameters: 
%   subj (must be input with single quotations, i.e., as a string!)
%   gsr - 1 if you would like to include a GSR regressor in the matrix, 0 if not   

    % Set up basic parameters
    basedir = ['/projects3/StroopActFlow/data/' subj];
    datadir = [basedir '/MNINonLinear/Results'];
    analysisdir = [basedir '/analysis'];
    numTasks = 6;
    trsPerRun = 745;
    totalTRs = trsPerRun * numTasks;
    numTaskStims = 8; % 8 regressors: 2 task rule encodings; 2 hand responses * 3 conditions (congruent, incongruent, neutral) 
    numMotionParams = 12; % HCP Pipe outputs 12
    trLength = .785; 
    
    %%
    % Need to create the derivative time series for ventricle, white matter, and whole brain signal
    disp(['Creating derivative time series for ventricle, white matter, and whole brain signal for subject ' subj])
    eval(['!1d_tool.py -overwrite -infile ' analysisdir '/' subj '_WM_timeseries_task.1D -derivative -write ' analysisdir '/' subj '_WM_timeseries_deriv_task.1D'])
    eval(['!1d_tool.py -overwrite -infile ' analysisdir '/' subj '_ventricles_timeseries_task.1D -derivative -write ' analysisdir '/' subj '_ventricles_timeseries_deriv_task.1D'])
    eval(['!1d_tool.py -overwrite -infile ' analysisdir '/' subj '_wholebrainsignal_timeseries_task.1D -derivative -write ' analysisdir '/' subj '_wholebrainsignal_timeseries_deriv_task.1D'])

    %%
    % Import movement regressors for subject, and reshape across all tasks
    % Create empty matrix for movement regressors for all tasks 
    % We will reshape these afterwards
    disp(['Importing movement regressors from HCP Pipeline Output for subject ' subj])
    movementRegressors = zeros(trsPerRun, numMotionParams, numTasks);
    for task=1:numTasks
        movementReg = [datadir '/Task' num2str(task) '/Movement_Regressors.txt'];
        movementRegressors(:,:,task) = importdata(movementReg);
    end

    movementRegressors = reshape(movementRegressors,[numMotionParams,totalTRs])'; 

    %%
    % Import timeseries regressors (ventricles, wm, derivatives, maybe whole brain?)
    % 6 in total, 2 for each wm, ventricles, and wholebrain
    disp(['Importing wm, ventricle and global brain time series into MATLAB for subj ' subj])
    timeseriesRegressors = zeros(totalTRs,6);
    % First 2 columns will be white matter
    timeseriesRegressors(:,1) = importdata([analysisdir '/' subj '_WM_timeseries_task.1D']);
    timeseriesRegressors(:,2) = importdata([analysisdir '/' subj '_WM_timeseries_deriv_task.1D']);
    % Columns 3 and 4 will be ventricles
    timeseriesRegressors(:,3) = importdata([analysisdir '/' subj '_ventricles_timeseries_task.1D']);
    timeseriesRegressors(:,4) = importdata([analysisdir '/' subj '_ventricles_timeseries_deriv_task.1D']);
    % Columns 5 and 6 will be whole brain signal, though we will opt to remove these out later if we do not want to perform GSR
    timeseriesRegressors(:,5) = importdata([analysisdir '/' subj '_wholebrainsignal_timeseries_task.1D']);
    timeseriesRegressors(:,6) = importdata([analysisdir '/' subj '_wholebrainsignal_timeseries_deriv_task.1D']);

    %%
    % Import task stim files
    stimMat = zeros(totalTRs,numTaskStims);
    for stim=1:numTaskStims
        stimdir = '/projects3/StroopActFlow/data/stimfiles/MotorResponseLocalizer/';
        stimfilename = dir([stimdir subj '*EV' num2str(stim) '_*']);
        stimMat(:,stim) = importdata([stimdir stimfilename.name]);
        taskStims = stimMat; % Bad coding, but this is because this script was adapted from previous scripts
    end

    %%
    % Convolve taskstim regressors based on SPM canonical HRF (likely period of task-induced activity)
    disp(['Convolving task stimulus binary files with SPM canonical HRF for subject ' subj])
    taskStims_HRF = zeros(size(taskStims));
    % Assuming .78
    spm_hrfTS = spm_hrf(trLength);
    for stim=1:size(taskStims,2)
        %% Convolve with canonical SPM HRF
        % Retrieve the task stimulus array for a particular task condition
        regressorBinaryEvents = taskStims(:,stim);
        taskR_reconvolved = zeros(1, length(regressorBinaryEvents));
        % Find the indices of which this particular task condition is occurring (the TR, or the row #)
        taskOnsets = find(regressorBinaryEvents);
        for i=1:length(taskOnsets)
            taskOnset = taskOnsets(i);
            taskR_reconvolved(taskOnset) = taskR_reconvolved(taskOnset) + spm_hrfTS(1);
            for lag=1:length(spm_hrfTS)-1
                if ((taskOnset+lag)<=length(taskR_reconvolved))
                    taskR_reconvolved(taskOnset+lag)=taskR_reconvolved(taskOnset+lag) + spm_hrfTS(lag+1);
                end
            end
        end
        taskStims_HRF(:,stim) = taskR_reconvolved';
    end
    

    %% 
    % if GSR is not selected (i.e., 0), then remove it from the timeseriesRegressor
    if gsr == 0 
        timeseriesRegressors = timeseriesRegressors(:,1:4);
    end


    %%
    % Create final output stim times for ALL regressors
    % Task regressors    
    linearTrend = 1:totalTRs;
    linearTrend = linearTrend';
    regressors = [linearTrend movementRegressors timeseriesRegressors taskStims_HRF];

    output.taskStims_HRF = taskStims_HRF;
    output.regressors = regressors;
    output.binaryTaskStims = taskStims;
end
    

    
