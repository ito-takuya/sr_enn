function refFuncDataStruct = takuReffunc_ModalityControl(EDATImport, analysisName)
%Taku Ito
%Reference Function converter from EDAT txt output to MATLAB EDAT struct
%Include RT reference at a later date.

EDATDataStruct = EDATImport;
%subjectNames = subjectArray;


%% Default values
numTRs_TaskRuns = 581;
%numTRsPerEDATLine = 6;
numProbesPerTrial = 3; %Is this the equivalent of the number of stimuli/responses for each instruction set?
numTrialsPerRun = 16;
numTaskRunsPerSubj = 8;

numTRsPerTrial = 36; 
numTRsForEncoding = 5; % Instruction screen?
numTRsPerProbe = 3;
TRvalue = 785;


for sNum = 1:length(EDATDataStruct)
    subjectName = EDATDataStruct{sNum}.SubjID;

    %Find Subject
    for thisSubjNum = 1:length(EDATDataStruct)
        if EDATDataStruct{thisSubjNum}.SubjID == subjectName
            subjNum = thisSubjNum;
        end
    end

    EDATvarHeader = EDATDataStruct{subjNum}.EDATData(1,:);
    EDATvar = EDATDataStruct{subjNum}.EDATData(11:end,:);       %Skip first 10 lines, since those are prac trial response)

    %First jitter
    firstJitterEDATAIndex = strmatch('blockITI', EDATvarHeader, 'exact');
    firstJitterEDATA = EDATvar(:, firstJitterEDATAIndex);

    %Second jitter
    secondJitterEDATAIndex = strmatch('InterBlockInterval[LogLevel5]', EDATvarHeader, 'exact');
    secondJitterEDATA = EDATvar(:, secondJitterEDATAIndex);

    %TrialType (repeat vs. novel)
    trialTypeEDATAIndex = strmatch('TaskType_rec', EDATvarHeader, 'exact');
    trialTypeEDATA = EDATvar(:, trialTypeEDATAIndex);

    %Delay between probes ** previously said 'add 2000ms to these values...
    probeDelayEDATAIndex = strmatch('probedelay[LogLevel6]', EDATvarHeader, 'exact');
    probeDelayEDATA = EDATvar(:, probeDelayEDATAIndex);

    %Feedback info 
    feedbackEDATAIndex = strmatch('Feedback[LogLevel6]',EDATvarHeader, 'exact');
    feedbackEDATA = EDATvar(:,feedbackEDATAIndex);

    %Logic cue info
    logicCueEDATAIndex = strmatch('LogicCue[LogLevel5]', EDATvarHeader,'exact');
    logicCueEDATA = EDATvar(:,logicCueEDATAIndex);

    %Semantic cue info
    semanticCueEDATAIndex = strmatch('SemanticCue[LogLevel5]', EDATvarHeader, 'exact');
    semanticCueEDATA = EDATvar(:,semanticCueEDATAIndex);

    %Response cue info
    responseCueEDATAIndex = strmatch('ResponseCue[LogLevel5]', EDATvarHeader,'exact');
    responseCueEDATA = EDATvar(:,responseCueEDATAIndex);

    taskidEDATAIndex = strmatch('TaskName[LogLevel5]', EDATvarHeader,'exact');
    taskidEDATA = EDATvar(:,taskidEDATAIndex);

    % Identifier to count miniblocks
    miniblockCount = 1;
    %%%%
    % Because there is a bug in the E-Prime script that does not correctly encode Novel/Practiced tasks (all tasks end up being encoded as novel), we will have to pull out the practiced tasks (which are thankfully labeled) in the EDAT file, and determine which tasks are practiced
%    pracTaskAEDATAIndex = strmatch('PracTaskA', EDATvarHeader, 'exact');
%    pracTaskAEDATA = EDATvar(:, pracTaskAEDATAIndex);
%
%    pracTaskBEDATAIndex = strmatch('PracTaskB', EDATvarHeader, 'exact');
%    pracTaskBEDATA = EDATvar(:, pracTaskBEDATAIndex);
%    
%    pracTaskCEDATAIndex = strmatch('PracTaskC', EDATvarHeader, 'exact');
%    pracTaskCEDATA = EDATvar(:, pracTaskCEDATAIndex);
%    
%    pracTaskDEDATAIndex = strmatch('PracTaskD', EDATvarHeader, 'exact');
%    pracTaskDEDATA = EDATvar(:, pracTaskDEDATAIndex);
    
    %% Modify task cue strings

    logicCueEDATA=regexprep(logicCueEDATA, '\*\*BOTH\*\*', 'BOTH');
    logicCueEDATA=regexprep(logicCueEDATA, '\*EITHER\*', 'EITHER');
    logicCueEDATA=regexprep(logicCueEDATA, 'NEITHER\*', 'NEITHER');
    logicCueEDATA=regexprep(logicCueEDATA, 'NOT\*BOTH', 'NOTBOTH');

    semanticCueEDATA=regexprep(semanticCueEDATA, '\*\*RED\*\*\*', 'RED');
    semanticCueEDATA=regexprep(semanticCueEDATA, 'CONSTANT', 'CONSTANT');
    semanticCueEDATA=regexprep(semanticCueEDATA, 'VERTICAL', 'VERTICAL');
    semanticCueEDATA=regexprep(semanticCueEDATA, 'HI\*PITCH', 'HIPITCH');

    responseCueEDATA=regexprep(responseCueEDATA, '\*LEFT\*INDEX\*', 'LINDEX');
    responseCueEDATA=regexprep(responseCueEDATA, 'RIGHT\*INDEX\*', 'RINDEX');
    responseCueEDATA=regexprep(responseCueEDATA, 'RIGHT\*MIDDLE', 'RMID');
    responseCueEDATA=regexprep(responseCueEDATA, 'LEFT\*MIDDLE\*', 'LMID');


    %% Creating reference function
    reffunc = cell(numTRs_TaskRuns*numTaskRunsPerSubj,1);
    msOffsetList = zeros(size(reffunc)); %Keeps track of timing error between TR and event onsets (should only be off by 1 second max!)
    msCount = 0;   %Keeps track of event onset in miliseconds (to track difference from TR onsets)
    currentLine = 1;
    currentLine_EDAT = 1;
    nonswitchTrialCount = 0;

    for runNum = 1:numTaskRunsPerSubj

        %% There is a run start delay!!
        for runStartDelayTRs = 1:5 %Hard coded this because unique to this study
            reffunc{currentLine} = ['Run' num2str(runNum) '_StartDelay'];
            msCount = msCount + TRvalue;
            msOffsetList(currentLine) = (currentLine * TRvalue) - msCount;
            currentLine = currentLine + 1;
        end

        for trialNum = 1:numTrialsPerRun

            disp(['Running task num: ' num2str(runNum) ' trialNum: ' num2str(trialNum) ' line num ' num2str(currentLine_EDAT)])
            %need to change trialType - trialType isn't correct, will need to figure out which trials are practiced for certain subjects and re-label them
            trialType = trialTypeEDATA{currentLine_EDAT};
            logicCue = logicCueEDATA{currentLine_EDAT};
            semanticCue = semanticCueEDATA{currentLine_EDAT};
            responseCue = responseCueEDATA{currentLine_EDAT};
            taskID = ['TaskNum' taskidEDATA{currentLine_EDAT}];
            mbID = ['Miniblock' num2str(miniblockCount)];        
            % Keep track of first mini-block line/TR
            startLine = currentLine;
            %Make sure nonswitch trials are marked
            if trialNum > 1
                thisTaskString = [logicCue semanticCue responseCue taskID];
                if strcmp(thisTaskString, lastTaskString)
                    nonswitchTrialCount = nonswitchTrialCount + 1;
                    for i = 1:numTRsPerTrial
                        reffunc{currentLine} = 'NonSwitchTrial';
                        msCount = msCount + TRvalue;
                        msOffsetList(currentLine) = (currentLine * TRvalue) - msCount;
                        currentLine = currentLine + 1;
                    end
                    currentLine_EDAT = currentLine_EDAT + 3;
                    lastTaskString = [logicCue semanticCue responseCue taskID];
                    continue
                end
            end
            lastTaskString = [logicCue semanticCue responseCue taskID];
 
            numTRsLeftForTrial = numTRsPerTrial;

            %Set encoding
            for encodingTR = 1:numTRsForEncoding
                if encodingTR == 1
                    reffunc{currentLine} = [trialType '_Task_Encoding_' logicCue '_' semanticCue '_' responseCue '_' taskID '_' mbID];
                else
                    reffunc{currentLine} = [trialType '_Task_Enc_' logicCue '_' semanticCue '_' responseCue '_' taskID '_' mbID];
                end
                
                msCount = msCount + TRvalue;
                numTRsLeftForTrial = numTRsLeftForTrial - 1;
                msOffsetList(currentLine) = (currentLine * TRvalue) - msCount;
                currentLine = currentLine + 1;
            end

            
            %Set first jittered delay 
            if rem(str2num(firstJitterEDATA{currentLine_EDAT}),TRvalue)==0
                firstJitterTRs = str2num(firstJitterEDATA{currentLine_EDAT})/TRvalue;
                for jitter1TR = 1:firstJitterTRs
                    if jitter1TR == 1
                        reffunc{currentLine} = [trialType '_' logicCue '_' semanticCue '_' responseCue '_' taskID '_' mbID '_StartRest'];
                    else
                        reffunc{currentLine} = [trialType '_' logicCue '_' semanticCue '_' responseCue '_' taskID '_' mbID '_Rest'];
                    end
                    numTRsLeftForTrial = numTRsLeftForTrial - 1;
                    msCount = msCount + TRvalue;
                    msOffsetList(currentLine) = (currentLine * TRvalue) - msCount;
                    currentLine = currentLine + 1;
                end
            else
                ['need to use floor/ceiling on first jitter delay! jitters dont align perfectly with TRs']
            end
            
            %Set probes & probe delays
            for probeNum=1:numProbesPerTrial
                probeDelay = str2num(probeDelayEDATA{currentLine_EDAT});

                %We know probe delay for this paradigm is always 1570ms = 2 TRs or 0ms (if 3rd probe)
                %This is just to see if we catch an error
                if probeDelay==0 || probeDelay==1570
                    numTRsForProbeDelay = 2;
                else
                    ['Error detected!']
                end

                probeAccuracy = feedbackEDATA{currentLine_EDAT};
                miniblock_good = 0;
                for probeTR = 1:numTRsPerProbe
                    if strcmp(probeAccuracy, 'Wrong')
                        reffunc{currentLine} = [trialType '_Task_Probe' num2str(probeNum) '_Wrong_' logicCue '_' semanticCue '_' responseCue '_' taskID '_' mbID];
                    else
                        reffunc{currentLine} = [trialType '_Task_Probe' num2str(probeNum) '_Correct_' logicCue '_' semanticCue '_' responseCue '_' taskID '_' mbID];
                        miniblock_good = miniblock_good + 1; % count how ever many probes were correct
                    end
                    
                    msCount = msCount + TRvalue;
                    numTRsLeftForTrial = numTRsLeftForTrial - 1;
                    msOffsetTrialCheck = (currentLine * TRvalue) - msCount;
                    msOffsetList(currentLine) = (currentLine * TRvalue) - msCount;
                    currentLine = currentLine + 1;
                end
                %If last probe, wait to increment EDAT line (needed for the 2nd jitter delay)
                if probeNum ~= numProbesPerTrial
                    for probeDelayTR = 1:numTRsForProbeDelay
                        reffunc{currentLine} = [trialType '_' logicCue '_' semanticCue '_' responseCue '_' taskID '_' mbID '_ProbeRest'];
                        numTRsLeftForTrial = numTRsLeftForTrial -1;
                        msOffsetList(currentLine) = (currentLine * TRvalue) - msCount;
                        currentLine = currentLine + 1;
                    end

                    currentLine_EDAT = currentLine_EDAT + 1;
                    msCount = msCount + probeDelay;
                end
            end

            %Set second jittered delay
            if rem(str2num(secondJitterEDATA{currentLine_EDAT}),TRvalue)==0
                secondJitterTRs = str2num(secondJitterEDATA{currentLine_EDAT})/TRvalue;
                for jitter2TR = 1:secondJitterTRs
                    if jitter2TR == secondJitterTRs
                        %reffunc{currentLine} = [trialType '_' logicCue '_' semanticCue '_' responseCue '_' taskID '_' mbID '_EndRest'];
                        reffunc{currentLine} = 'EndRest';
                    else
                        %reffunc{currentLine} = [trialType '_' logicCue '_' semanticCue '_' responseCue '_' taskID '_' mbID '_Rest'];
                        reffunc{currentLine} = 'Rest';
                    end
                    numTRsLeftForTrial = numTRsLeftForTrial - 1;
                    msCount = msCount + TRvalue;
                    msOffsetList(currentLine) = (currentLine * TRvalue) - msCount;
                    currentLine = currentLine + 1;
                end

                %Still on the EDAT line for the third/last probe, so go to next EDAT Line
                currentLine_EDAT = currentLine_EDAT + 1;
                %Count next miniblock
                miniblockCount = miniblockCount + 1;
            else
                ['need to use floor/ceiling on first jitter delay! jitters dont align perfectly with TRs']
            end

            % Now go back and evaluate if it was a good mini-block (good = 2/3 probes correct, bad otherwise)
            % Exclude the second jitter
            for trsPerMiniBlock=1:(numTRsPerTrial-secondJitterTRs)
                if miniblock_good >= 2
                    reffunc{startLine} = [reffunc{startLine} '_MiniblockGood'];
                else
                    reffunc{startLine} = [reffunc{startLine} '_MiniblockBad'];
                    ['bad exists ' num2str(startLine)]
                end
                startLine = startLine + 1;
            end
                    

            % Now do some sanity checking
            msOffsetTrialCheck = ((currentLine-1) * TRvalue) - msCount; %-1 currentLine since we just added it.
            if abs(msOffsetTrialCheck) > 0
                disp(['Warning: There is an offset at the end of a trial (' num2str(secondsOffsetTrialCheck) ' second(s))']);
            end

            if numTRsLeftForTrial ~=0
                disp(['Warning: TR Count doesn''t match at the end of trial (' num2str(numTRsLeftForTrial) ' left']);

            end

            disp(['For Run Number ' num2str(runNum) ', the Trial offset for ms is: ' num2str(msOffsetTrialCheck)]);
            disp(['For Run Number ' num2str(runNum) ', the number of TRs left is: ' num2str(numTRsLeftForTrial)]);
        end

        disp(['msCounts = ' num2str(msCount)]);
    
        if ~isempty(find(abs(msOffsetList) > 1))
            disp(['Warning: TR offsets are occasionally greater than 1 for reference function']);
        end

        refFuncDataStruct{subjNum} = EDATDataStruct{subjNum};
        refFuncDataStruct{subjNum}.RefFunc = reffunc;
        refFuncDataStruct{subjNum}.AnalysisName = analysisName;
        refFuncDataStruct{subjNum}.RefFuncTimeOffsetList = msOffsetList;
        refFuncDataStruct{subjNum}.nonswitchTrialCount = nonswitchTrialCount;
    
    end

end

















































