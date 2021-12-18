function output = loadStimFiles_inputStim(subj, gsr, inputtype)
% Taku Ito
% 3/12/15
%
% This script imports the stim files for IndivRITL modality control into a regressor matrix including wm/ventricle/wholebrain timeseries, 12 motion parameters, and 8 stimulus time series
% Regressing out WM, Ventricles (and corresponding derivatives), motion parameters, and auditory versus visual rule contrasts (i.e., constant and high pitch versus vertical and red)
% 
% Parameters: 
%   subj (must be input with single quotations, i.e., as a string!)
%   gsr - 1 if you would like to include a GSR regressor in the matrix, 0 if not   
%   inputtype = ['Color', 'Ori', 'Pitch', 'Constant']

    % Set up basic parameters
    basedir = ['/projects/IndivRITL/data/' subj];
    datadir = [basedir '/MNINonLinear/Results'];
    analysisdir = [basedir '/analysis'];
    numTasks = 8;
    totalTRs = 4648;
    totalRestTRs = 1070; 
    trsPerRun = 581;
    numTaskStims = 4; % 4 motor outputs
    numMotionParams = 12; % HCP Pipe outputs 12
    trLength = .785; 
    
    %%
    % Need to create the derivative time series for ventricle, white matter, and whole brain signal
    disp(['Creating derivative time series for ventricle, white matter, and whole brain signal for subject ' subj])
    eval(['!1d_tool.py -overwrite -infile ' analysisdir '/' subj '_WM_timeseries_task.1D -derivative -write ' analysisdir '/' subj '_WM_timeseries_deriv_task.1D'])
    eval(['!1d_tool.py -overwrite -infile ' analysisdir '/' subj '_ventricles_timeseries_task.1D -derivative -write ' analysisdir '/' subj '_ventricles_timeseries_deriv_task.1D'])
    eval(['!1d_tool.py -overwrite -infile ' analysisdir '/' subj '_wholebrainsignal_timeseries_task.1D -derivative -write ' analysisdir '/' subj '_wholebrainsignal_timeseries_deriv_task.1D'])
    eval(['!1d_tool.py -overwrite -infile ' analysisdir '/' subj '_WM_timeseries_rest.1D -derivative -write ' analysisdir '/' subj '_WM_timeseries_deriv_rest.1D'])
    eval(['!1d_tool.py -overwrite -infile ' analysisdir '/' subj '_ventricles_timeseries_rest.1D -derivative -write ' analysisdir '/' subj '_ventricles_timeseries_deriv_rest.1D'])
    eval(['!1d_tool.py -overwrite -infile ' analysisdir '/' subj '_wholebrainsignal_timeseries_rest.1D -derivative -write ' analysisdir '/' subj '_wholebrainsignal_timeseries_deriv_rest.1D'])

    %%
    % Import movement regressors for subject, and reshape across all tasks
    % Create empty matrix for movement regressors for all tasks (581TRs x 12 parameters x 8 runs)
    % We will reshape these afterwards
    disp(['Importing movement regressors from HCP Pipeline Output for subject ' subj])
    movementRegressors = zeros(trsPerRun, numMotionParams, numTasks);
    for task=1:numTasks
        movementReg = [datadir '/Task' num2str(task) '/Movement_Regressors.txt'];
        movementRegressors(:,:,task) = importdata(movementReg);
    end

    movementRegressors = reshape(movementRegressors,[numMotionParams,totalTRs])'; %This will create a 4648 x 12 matrix (note the transpose)

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

    %% Create linear trend regressor for each run
    linearTrendRegs = zeros(totalTRs,numTasks);
    trstart = 1;
    for run=1:numTasks
        trend = trstart + trsPerRun - 1;
        linearTrendRegs(trstart:trend,run) = 1:trsPerRun;
        trstart = trstart + trsPerRun;
    end

    %%
    % Import task stim files
    % There are 8 for this particular GLM - 4 for each sensory rule * 2 for correct V error performance
    disp(['Importing ' num2str(numTaskStims) ' task stimulus timing files for subject ' subj])
    taskStims = zeros(totalTRs,numTaskStims);
    for stim=1:numTaskStims
        stimdir = ['/projects3/SRActFlow/data/stimfiles/cpro' inputtype 'Stim/'];
        stimFileName = dir([stimdir '/' subj '*EV' num2str(stim) '*']);
        taskStims(:,stim) = importdata([stimdir '/' stimFileName.name]);
    end
%    stimMat = zeros(totalTRs,numTaskStims);
%    for stim=1:numTaskStims
%        stimMat(:,stim) = importdata([stimdir stimfilename.name]);
%        taskStims = stimMat; % Bad coding, but this is because this script was adapted from previous scripts
%    end

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
    regressors = [linearTrendRegs movementRegressors timeseriesRegressors taskStims_HRF];

    % If we want to regress out 

    output.taskStims_HRF = taskStims_HRF;
    output.regressors = regressors;
    output.binaryTaskStims = taskStims;
end
    

    
