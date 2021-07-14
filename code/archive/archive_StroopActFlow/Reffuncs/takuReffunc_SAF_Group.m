% Taku Ito
% 01/12/2017

% This is a general script to import EDATs into MATLAB and produce stim files according to the type of stimulus files you want.
% Subject numbers - this will need to incrementally edited as more subjects become preprocessed
subjNums = [101 102]; 
nTrials = 540;
nMiniblocks = 60;
% Need to create this for ReffuncToAFNI function
subjNumStr = '101 102';

subjNumStr = strread(subjNumStr, '%s', 'delimiter', ' ');

% Import fMRI Behavioral EDAT files
EDATImport = taku_EDATImportBySubj_SAF('_fMRI_SAF.txt', subjNums);

% Run reference function, converting trial information to TR-by-TR info
reffunc = takuReffunc_SAF(EDATImport, 'SAF');

% Put all subjects into a single vector
reffunc_vector = takuReffuncVector_SAF(reffunc);

%% Reffunc for TrialTypesXCondition
% ReffuncToAFNI - create stimulus timingfiles in the current directory
execute = 0;
if execute == 1
    disp('Running Reffunc for TrialTypes X Condition')
    
    [dmat, dLabels] = ReffuncToAFNI(reffunc_vector, subjNumStr, 6, [1:6], [745 745 745 745 745 745], .785,...
        {'NeutralColor', 'NeutralWord', 'CongruentColor', 'CongruentWord', 'IncongruentColor', 'IncongruentWord'},...
        {'\w*_Block.color_Trial.neutral_\w*', '\w*_Block.word_Trial.neutral_\w*',...
        '\w*_Block.color_Trial.compatible_\w*', '\w*_Block.word_Trial.compatible_\w*',...
        '\w*_Block.color_Trial.incompatible_\w*', '\w*_Block.word_Trial.incompatible_\w*'},...
        1, 1, 'TrialTypesXCondition')
end


%% Reffunc for rule encodings (i.e., top-down control) -- include instructions + post-instruction delay
execute = 0;
if execute == 1
    disp('Running Reffunc for Rule Encodings')

    [dmat, dLabels] = ReffuncToAFNI(reffunc_vector, subjNumStr, 6, [1:6], [745 745 745 745 745 745], .785,...
        {'ColorEncoding', 'WordEncoding'},...
        {'\w*_color_\w*Encoding\w*', '\w*_word_\w*Encoding\w*'},...
        1, 1, 'TaskRuleEncodings')
end


%% Reffunc for encoding, stim, response (all in one model)
execute = 0;
if execute == 1
    disp('Running beta series for encoding and stimulus all in one model')

    % Construct condition labeling
    i = 1;
    for mb=1:nMiniblocks
        conditionLabel{i} = ['RuleEncoding_' num2str(mb)];
        conditionSearchStr{i} = ['\w*Encoding\w*Miniblock' num2str(mb) '_'];
        i = i + 1;
    end
    for trial=1:nTrials
        conditionLabel{i} = ['Trial_' num2str(trial)];
        conditionSearchStr{i} = ['\w*_Trial' num2str(trial) '_\w*'];
        i = i + 1;
    end
    % No regular expression! Only exact matches.
    [dmat, dLabels] = ReffuncToAFNI(reffunc_vector, subjNumStr, 6, [1:6], [745 745 745 745 745 745], .785, conditionLabel, conditionSearchStr, ...
        1, 1, 'RuleStimEncoding')
end

%% Output behavioral responses to CSVs
execute = 0;
if execute == 1
    disp('Output behavioral data to CSVs for analysis purposes')

    outdir = '/projects3/StroopActFlow/data/results/behavdata/';

    % Extract information for each trial
    outputBehavData(EDATImport, outdir)

    for subj=1:length(subjNums)
        % Now extract information for each miniblock (i.e., which rule is associated with each miniblock)
        i = 1;
        for mb=1:nMiniblocks
            conditionSearchColor{i} = ['\w*_color_TaskEncoding\w*_Miniblock' num2str(mb) '_'];
            conditionSearchWord{i} = ['\w*_word_TaskEncoding\w*_Miniblock' num2str(mb) '_'];
            i = i + 1;
        end

        tr = 1;
        mb = 1;
        while mb<=nMiniblocks
            if regexp(reffunc{subj}.RefFunc{tr},conditionSearchColor{mb})
                mb_rule{mb} = 'Color';
                mb = mb + 1;
                tr = tr + 10; % Skip 10 TRs to ensure you go to next block
            elseif regexp(reffunc{subj}.RefFunc{tr},conditionSearchWord{mb})
                mb_rule{mb} = 'Word';
                mb = mb + 1;
                tr = tr + 10;
            else
                tr = tr + 1;
            end
        end
        outfile = [outdir num2str(subjNums(subj)) '_miniblockRuleEncodings.csv'];
        cell2csv(outfile, mb_rule, ',');
    end

end

%% Reffunc for motor responses (i.e., behavior) 
execute = 0;
if execute == 1
    disp('Running Reffunc for Rule Encodings')

    [dmat, dLabels] = ReffuncToAFNI(reffunc_vector, subjNumStr, 6, [1:6], [745 745 745 745 745 745], .785,...
        {'ColorEncoding', 'WordEncoding',...
        'LeftHandCongruent', 'RightHandCongruent', ...
        'LeftHandIncongruent', 'RightHandIncongruent', ...
        'LeftHandNeutral', 'RightHandNeutral'},...
        {'\w*_color_\w*Encoding\w*', '\w*_word_\w*Encoding\w*',...
        '\S*_compatible_\S*_Resp.y_\w*', '\S*_compatible_\S*_Resp.g_\w*',...
        '\S*_incompatible_\S*_Resp.y_\w*', '\S*_incompatible_\S*_Resp.g_\w*',...
        '\S*_neutral_\S*_Resp.y_\w*', '\S*_neutral_\S*_Resp.g_\w*'},...
        1, 1, 'MotorResponses')

%     [dmat, dLabels] = ReffuncToAFNI(reffunc_vector, subjNumStr, 6, [1:6], [745 745 745 745 745 745], .785,...
%         {'LeftHand', 'RightHand'},...
%         {'\w*_Resp.y_\w*', '\w*_\w*_Resp.g_\w*'},...
%         1, 1, 'MotorResponses')
end

%% Reffunc for hidden layer regions 
execute = 0;
if execute == 1
    disp('Running Reffunc for Hidden layer nodes (S-R associations by task rule)')

    [dmat, dLabels] = ReffuncToAFNI(reffunc_vector, subjNumStr, 6, [1:6], [745 745 745 745 745 745], .785,...
        {'NeutralColor_Green', 'NeutralColor_Red', 'NeutralWord_Green', 'NeutralWord_Red'},...
        {'\S*_Block.color_\w*_neutral_colorstim.green_wordstim.XXXX_\w*_Resp.g_\w*', ...
        '\S*_Block.color_\w*_neutral_colorstim.red_wordstim.XXXX_\w*_Resp.y_\w*', ...
        '\S*_Block.word_\w*_neutral_colorstim.black_wordstim.GREEN_\w*_Resp.g_\w*', ...
        '\S*_Block.word_\w*_neutral_colorstim.black_wordstim.RED_\w*_Resp.y_\w*'}, ...
        1, 1, 'HiddenLayerLocalization')



end

%% Reffunc for localizers -- Neutral stimuli 
execute = 1;
if execute == 1
    disp('Running Reffunc for Neutral Stimuli Localizers')

    % Note: Congruent Color Stim Green == Congruent Word Stim Green
    % Note: Congruent Color Stim Red == Congruent Word Stim Red
    % Note: Incongruent Color Stim Green == Incongruent Word Stim Red
    % Note: Incongruent Color Stim Red == Incongruent Word Stim Green
    % So 10 orthogonal regressors
    [dmat, dLabels] = ReffuncToAFNI(reffunc_vector, subjNumStr, 6, [1:6], [745 745 745 745 745 745], .785,...
        {'ColorEncoding', 'WordEncoding',...
        'CongruentStimGreen', 'CongruentStimRed', ...
        'IncongruentColorStimGreen', 'IncongruentColorStimRed',...
        'NeutralColorStimGreen', 'NeutralColorStimRed', 'NeutralWordStimGreen', 'NeutralWordStimRed'},...
        {'\w*_color_\w*Encoding\w*', '\w*_word_\w*Encoding\w*',...
        '\S*_compatible_colorstim.green_\S*', '\S*_compatible_colorstim.red_\S*',...
        '\S*_incompatible_colorstim.green_\S*', '\S*_incompatible_colorstim.red_\S*',...
        '\S*_neutral_colorstim.green_\S*', '\S*_neutral_colorstim.red_\S*',...
        '\S*_neutral_\S*_wordstim.GREEN_\S*', '\S*_neutral_\S*_wordstim.RED_\S*'},...
        1, 1, 'NeutralStimuli_Localizers')
end
