function outputBehavData(EDATImport, outputdir)
%Taku Ito
% 02/21/2017
%Reference Function converter from EDAT txt output to MATLAB EDAT struct

EDATDataStruct = EDATImport;
%subjectNames = subjectArray;


%% Default values
TRvalue = 785;

numTRs_TaskRuns = 745;
numTRsPerEDATLine = 2;
numTrialsPerBlock = 18; % Includes inter-trial delay 'trials'
numBlocksPerRun = 10;
numTaskRunsPerSubj = 6;

numTRsPerBlock = 74; 
numTRsPerTrial = 2;

numTRsForEncoding = 2; % Instruction screen?
numTRsForEncodingDelay = 5;
numTRsRunStartDelay = 5;
numTRsPostBlockDelay = 31;



for sNum = 1:length(EDATDataStruct)
    subjectName = EDATDataStruct{sNum}.SubjID;

    fileprefix = [outputdir subjectName '_'];
    EDATvarHeader = EDATDataStruct{sNum}.EDATData(1,:);
    EDATvar = EDATDataStruct{sNum}.EDATData(2:end,:);       %Skip first line, since it's EDAT header

%    %First jitter
%    firstJitterEDATAIndex = strmatch('blockITI', EDATvarHeader, 'exact');
%    firstJitterEDATA = EDATvar(:, firstJitterEDATAIndex);
%
%    %Second jitter
%    secondJitterEDATAIndex = strmatch('InterBlockInterval[LogLevel5]', EDATvarHeader, 'exact');
%    secondJitterEDATA = EDATvar(:, secondJitterEDATAIndex);

    %TrialType (neutral v. compatible v. incompatible v. delay)
    trialTypeEDATAIndex = strmatch('Condition', EDATvarHeader, 'exact');
    trialTypeEDATA = EDATvar(:, trialTypeEDATAIndex);

    % Make sure we only include actual trials (i.e., not delay periods)
    count = 1;
    for i=1:length(trialTypeEDATA)
        if ~strcmp(trialTypeEDATA{i},'delay');
            ind(count) = i;
            count = count + 1;
        end
    end
    cell2csv([fileprefix 'condition.csv'],trialTypeEDATA(ind),',')

    %Delay between probes ** previously said 'add 2000ms to these values...
%    probeDelayEDATAIndex = strmatch('probedelay[LogLevel6]', EDATvarHeader, 'exact');
%    probeDelayEDATA = EDATvar(:, probeDelayEDATAIndex);

    %Feedback info 
    feedbackEDATAIndex = strmatch('Stimulus.ACC',EDATvarHeader, 'exact');
    feedbackEDATA = EDATvar(:,feedbackEDATAIndex);
    cell2csv([fileprefix 'accuracy.csv'],feedbackEDATA(ind),',')

    %Subj response info
    subjResponseEDATAIndex = strmatch('Stimulus.RESP',EDATvarHeader, 'exact');
    subjResponseEDATA = EDATvar(:,subjResponseEDATAIndex);
    % Make no responses have 'noresponse' string
    for trial=1:length(ind)
        if strcmp(subjResponseEDATA(ind(trial)),'y') || strcmp(subjResponseEDATA(ind(trial)),'g')
            responsearray{trial} = subjResponseEDATA{ind(trial)};
        else
            responsearray{trial} = 'noresponse';
        end
    end
    cell2csv([fileprefix 'subjResponse.csv'],responsearray,',')

    %Task rule
    taskRuleEDATAIndex = strmatch('Task[Trial]', EDATvarHeader,'exact');
    taskRuleEDATA = EDATvar(:,taskRuleEDATAIndex);
    cell2csv([fileprefix 'taskRule.csv'],taskRuleEDATA(ind),',')

    %Color stim
    colorStimEDATAIndex = strmatch('ColorStim', EDATvarHeader, 'exact');
    colorStimEDATA = EDATvar(:,colorStimEDATAIndex);
    cell2csv([fileprefix 'colorStim.csv'],colorStimEDATA(ind),',')

    %Word stim
    wordStimEDATAIndex = strmatch('WordStim', EDATvarHeader,'exact');
    wordStimEDATA = EDATvar(:,wordStimEDATAIndex);
    cell2csv([fileprefix 'wordStim.csv'],wordStimEDATA(ind),',')

    % Block number (runs are referred to as blocks)
    runNumEDATAIndex = strmatch('Block', EDATvarHeader,'exact');
    runNumEDATA = EDATvar(:,runNumEDATAIndex);

    %  Block number (blocks are referred to as trials in EDATA)
    blockNumEDATAIndex = strmatch('Trial', EDATvarHeader,'exact');
    blockNumEDATA = EDATvar(:,blockNumEDATAIndex);
    
    % RT
    rtEDATAIndex = strmatch('Stimulus.RT', EDATvarHeader,'exact');
    rtEDATA = EDATvar(:,rtEDATAIndex);
    cell2csv([fileprefix 'RT.csv'],rtEDATA(ind),',')
end
