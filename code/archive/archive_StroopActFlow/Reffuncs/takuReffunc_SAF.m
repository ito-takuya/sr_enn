function refFuncDataStruct = takuReffunc_SAF(EDATImport, analysisName)
%Taku Ito
% 1/12/2017
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

    %Find Subject
    for thisSubjNum = 1:length(EDATDataStruct)
        if EDATDataStruct{thisSubjNum}.SubjID == subjectName
            subjNum = thisSubjNum;
        end
    end

    EDATvarHeader = EDATDataStruct{subjNum}.EDATData(1,:);
    EDATvar = EDATDataStruct{subjNum}.EDATData(2:end,:);       %Skip first line, since it's EDAT header

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

    %Delay between probes ** previously said 'add 2000ms to these values...
%    probeDelayEDATAIndex = strmatch('probedelay[LogLevel6]', EDATvarHeader, 'exact');
%    probeDelayEDATA = EDATvar(:, probeDelayEDATAIndex);

    %Feedback info 
    feedbackEDATAIndex = strmatch('Stimulus.ACC',EDATvarHeader, 'exact');
    feedbackEDATA = EDATvar(:,feedbackEDATAIndex);

    %Subj response info
    subjResponseEDATAIndex = strmatch('Stimulus.RESP',EDATvarHeader, 'exact');
    subjResponseEDATA = EDATvar(:,subjResponseEDATAIndex);

    %Task rule
    taskRuleEDATAIndex = strmatch('Task[Trial]', EDATvarHeader,'exact');
    taskRuleEDATA = EDATvar(:,taskRuleEDATAIndex);

    %Color stim
    colorStimEDATAIndex = strmatch('ColorStim', EDATvarHeader, 'exact');
    colorStimEDATA = EDATvar(:,colorStimEDATAIndex);

    %Word stim
    wordStimEDATAIndex = strmatch('WordStim', EDATvarHeader,'exact');
    wordStimEDATA = EDATvar(:,wordStimEDATAIndex);

    % Block number (runs are referred to as blocks)
    runNumEDATAIndex = strmatch('Block', EDATvarHeader,'exact');
    runNumEDATA = EDATvar(:,runNumEDATAIndex);

    %  Block number (blocks are referred to as trials in EDATA)
    blockNumEDATAIndex = strmatch('Trial', EDATvarHeader,'exact');
    blockNumEDATA = EDATvar(:,blockNumEDATAIndex);
    
    % RT
    rtEDATAIndex = strmatch('Stimulus.RT', EDATvarHeader,'exact');
    rtEDATA = EDATvar(:,rtEDATAIndex);

    % Identifier to count miniblocks
    miniblockCount = 1;
    
    %% Modify task cue strings

%    logicCueEDATA=regexprep(logicCueEDATA, '\*\*BOTH\*\*', 'BOTH');
%    logicCueEDATA=regexprep(logicCueEDATA, '\*EITHER\*', 'EITHER');
%    logicCueEDATA=regexprep(logicCueEDATA, 'NEITHER\*', 'NEITHER');
%    logicCueEDATA=regexprep(logicCueEDATA, 'NOT\*BOTH', 'NOTBOTH');
%
%    semanticCueEDATA=regexprep(semanticCueEDATA, '\*\*RED\*\*\*', 'RED');
%    semanticCueEDATA=regexprep(semanticCueEDATA, 'CONSTANT', 'CONSTANT');
%    semanticCueEDATA=regexprep(semanticCueEDATA, 'VERTICAL', 'VERTICAL');
%    semanticCueEDATA=regexprep(semanticCueEDATA, 'HI\*PITCH', 'HIPITCH');
%
%    responseCueEDATA=regexprep(responseCueEDATA, '\*LEFT\*INDEX\*', 'LINDEX');
%    responseCueEDATA=regexprep(responseCueEDATA, 'RIGHT\*INDEX\*', 'RINDEX');
%    responseCueEDATA=regexprep(responseCueEDATA, 'RIGHT\*MIDDLE', 'RMID');
%    responseCueEDATA=regexprep(responseCueEDATA, 'LEFT\*MIDDLE\*', 'LMID');


    %% Creating reference function
    reffunc = cell(numTRs_TaskRuns*numTaskRunsPerSubj,1);
    msOffsetList = zeros(size(reffunc)); %Keeps track of timing error between TR and event onsets (should only be off by 1 second max!)
    msCount = 0;   %Keeps track of event onset in miliseconds (to track difference from TR onsets)
    currentLine = 1;
    currentLine_EDAT = 1;
    nonswitchTrialCount = 0;
    nonswitchBlockCount = 0;
    trialCount = 1;
    
    for runNum = 1:numTaskRunsPerSubj

        % There is a run start delay!!
        for runStartDelayTRs = 1:numTRsRunStartDelay 
            reffunc{currentLine} = ['Run' num2str(runNum) '_StartDelay'];
            msCount = msCount + TRvalue;
            msOffsetList(currentLine) = (currentLine * TRvalue) - msCount;
            currentLine = currentLine + 1;
        end

        for blockNum = 1:numBlocksPerRun

            disp(['Running task num: ' num2str(runNum) ' blockNum: ' num2str(blockNum) ' line num ' num2str(currentLine_EDAT)])
            
            % Get task rule for this block
            blockType = taskRuleEDATA{currentLine_EDAT};
            
            mbID = ['Miniblock' num2str(miniblockCount) '_'];        
            % Keep track of first mini-block line/TR
            startLine = currentLine;
            
            % Check if this task block is a switch block
            % Make sure nonswitch trials are marked
            if blockNum == 1
                switchBlock = 'switchBlock';
            else
                thisBlockString = blockType;
                if strcmp(thisBlockString, lastBlockString)
                    nonswitchBlockCount = nonswitchBlockCount + 1;
                    switchBlock = 'nonSwitchBlock';
                else
                    switchBlock = 'switchBlock';
                end
            end
            lastBlockString = blockType;

            
            numTRsLeftForBlock = numTRsPerBlock;

            %Set encoding
            for encodingTR = 1:numTRsForEncoding
                reffunc{currentLine} = [switchBlock '_' blockType '_TaskEncoding_' mbID];
                
                msCount = msCount + TRvalue;
                numTRsLeftForBlock = numTRsLeftForBlock - 1;
                msOffsetList(currentLine) = (currentLine * TRvalue) - msCount;
                currentLine = currentLine + 1;
            end

            %Set encoding delay
            for delayTR = 1:numTRsForEncodingDelay
                reffunc{currentLine} = [switchBlock '_' blockType '_EncodingDelay_' mbID];
                
                msCount = msCount + TRvalue;
                numTRsLeftForBlock = numTRsLeftForBlock - 1;
                msOffsetList(currentLine) = (currentLine * TRvalue) - msCount;
                currentLine = currentLine + 1;
            end
            
            % Begin block (trials within block)
            switchTrial = 'null';
            lastTrialString = '';
            for trialNum = 1:numTrialsPerBlock
                
                trialType = trialTypeEDATA{currentLine_EDAT};
                accuracy = feedbackEDATA{currentLine_EDAT};
                subjResponse = subjResponseEDATA{currentLine_EDAT};
                colorStim = colorStimEDATA{currentLine_EDAT};
                wordStim = wordStimEDATA{currentLine_EDAT};
                rtData = rtEDATA{currentLine_EDAT};
                
                % Identify delays versus 'real trial' (i.e., has stimulus)
                if strcmp(trialType, 'delay')
                    % Set condition for delay

                    for trialTR = 1:numTRsPerTrial
                        reffunc{currentLine} = [switchBlock '_' blockType '_ITI_' mbID];
                        
                        msCount = msCount + TRvalue;
                        numTRsLeftForBlock = numTRsLeftForBlock - 1;
                        msOffsetList(currentLine) = (currentLine * TRvalue) - msCount;
                        currentLine = currentLine + 1;
                    end
                    
                    currentLine_EDAT = currentLine_EDAT + 1;
                else
                    % Set condition for trial
                    
                    % Set accuracy string
                    if strcmp(accuracy,'1')
                        AccStr = 'Correct';
                    else
                        AccStr = 'Wrong';
                    end
                    
                    % Determine if this is a 'switch' trial
                    if strcmp(switchTrial, 'null')
                        % If null, it is the first trial of the block - so
                        % nonSwitchTrial
                        switchTrial = 'switchTrial';
                    else
                        thisTrialString = trialType;
                        if strcmp(thisTrialString, lastTrialString)
                            nonswitchTrialCount = nonswitchTrialCount + 1;
                            switchTrial = 'nonSwitchTrial';
                        else
                            switchTrial = 'switchTrial';
                        end
                        lastTrialString = trialType;
                    end
                
                    % Fill in reffunc
                    for trialTR = 1:numTRsPerTrial
                        reffunc{currentLine} = [switchBlock '_' switchTrial '_Block.' blockType '_Trial' num2str(trialCount) '_' trialType '_colorstim.' colorStim '_wordstim.' wordStim '_' AccStr '_Resp.' subjResponse '_RT.' num2str(rtData) '_' mbID];

                        msCount = msCount + TRvalue;
                        numTRsLeftForBlock = numTRsLeftForBlock - 1;
                        msOffsetList(currentLine) = (currentLine * TRvalue) - msCount;
                        currentLine = currentLine + 1;
                    end
                    trialCount = trialCount + 1;

                    currentLine_EDAT = currentLine_EDAT + 1;
                end
            end
                    
            % Now take care of post block delay
            for postBlockDelayTR = 1:numTRsPostBlockDelay
                reffunc{currentLine} = [switchBlock '_postBlockDelay'];
                
                msCount = msCount + TRvalue;
                numTRsLeftForBlock = numTRsLeftForBlock - 1;
                msOffsetList(currentLine) = (currentLine * TRvalue) - msCount;
                currentLine = currentLine + 1;   
            end

            % Now do some sanity checking
            msOffsetBlockCheck = ((currentLine-1) * TRvalue) - msCount; %-1 currentLine since we just added it.
            if abs(msOffsetBlockCheck) > 0
                disp(['Warning: There is an offset at the end of a trial (' num2str(secondsOffsetBlockCheck) ' second(s))']);
            end

            if numTRsLeftForBlock ~=0
                disp(['Warning: TR Count doesn''t match at the end of trial (' num2str(numTRsLeftForBlock) ' left']);

            end

            disp(['For Run Number ' num2str(runNum) ', the Trial offset for ms is: ' num2str(msOffsetBlockCheck)]);
            disp(['For Run Number ' num2str(runNum) ', the number of TRs left is: ' num2str(numTRsLeftForBlock)]);
            
            % Count the block number
            miniblockCount = miniblockCount + 1;
        end

        disp(['msCounts = ' num2str(msCount)]);
    
        if ~isempty(find(abs(msOffsetList) > 1))
            disp(['Warning: TR offsets are occasionally greater than 1 for reference function']);
        end
        
        disp(['Number of non-switch blocks: ' num2str(nonswitchBlockCount)])
        disp(['Number of non-switch trials: ' num2str(nonswitchTrialCount)])
        
        refFuncDataStruct{subjNum} = EDATDataStruct{subjNum};
        refFuncDataStruct{subjNum}.RefFunc = reffunc;
        refFuncDataStruct{subjNum}.AnalysisName = analysisName;
        refFuncDataStruct{subjNum}.RefFuncTimeOffsetList = msOffsetList;
        refFuncDataStruct{subjNum}.nonswitchTrialCount = nonswitchTrialCount;
    
    end

end

















































