% Taku Ito
% 3/12/15

% This is a general script to import EDATs into MATLAB and produce stim files according to the type of stimulus files you want.
% Originally created to create stimulus files for auditory v visual visual task rule contrasts, i.e., constant/hi-pitch versus vertical/red sensory task rules
addpath('../scripts/');
% Subject numbers - this will need to incrementally edited as more subjects become preprocessed
subjNums = [013 014 016 017 018 021 023 024 025 026 027 028 030 031 032 033 034 035 037 038 039 040 042 043 045 046 047 048 049 050 053 055 057 062]; % 063 066 067 068 069 070 072 074 075 077 081 086 088]; %041 
% excluding subject 041
% Need to create this for ReffuncToAFNI function
subjNumStr = '013 014 016 017 018 021 023 024 025 026 027 028 030 031 032 033 034 035 037 038 039 040 042 043 045 046 047 048 049 050 053 055 057 062';% 063 066 067 068 069 070 072 074 075 077 081 086 088'; %041
subjNumStr = strread(subjNumStr, '%s', 'delimiter', ' ');

% Import fMRI Behavioral EDAT files
EDATImport = taku_EDATImportBySubj_IndivRITL('_fMRI_CPRO.txt', subjNums);

% Run reference function, converting trial information to TR-by-TR info
reffunc = takuReffunc_ModalityControl_v3(EDATImport, 'modControl');

% Put all subjects into a single vector
reffunc_vector = takuReffuncVector_IndivRITL(reffunc);

% ReffuncToAFNI - create stimulus timingfiles in the current directory
[dmat, dLabels] = ReffuncToAFNI(reffunc_vector, subjNumStr, 8, [1:8], [581 581 581 581 581 581 581 581], .785,...
    {'CONSTANT', 'HIPITCH', 'VERTICAL', 'RED', ...
     'BOTH', 'NOTBOTH', 'EITHER', 'NEITHER',...
     'LMID', 'LINDEX', 'RMID', 'RINDEX'},...
    {'\w*_CONSTANT_\w*', '\w*_HIPITCH_\w*','\w*_VERTICAL_\w*', '\w*_RED_\w*',...
    '\w*_BOTH_\w*', '\w*_NOTBOTH_\w*','\w*_EITHER_\w*', '\w*_NEITHER_\w*',...
    '\w*_LMID_\w*', '\w*_LINDEX_\w*','\w*_RMID_\w*', '\w*_RINDEX_\w*',},...
    1, 1, 'CPRORules_ModalityControlv3')

