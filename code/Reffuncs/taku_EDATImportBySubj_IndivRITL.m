function EDATImport = EDATImportBySubject_IndivRITL_v1a(importFileEndString, subjNums)
% This function imports the EDAT file (that was previously exported from the .edat2 file to a txt file) for each individual subject into MATLAB
% Parameters: 
%   importFileEndString: This should be a string to indicate the end of the filename for each subject's exported file (should be a .txt file)
%                        e.g., if the file name for a subject is '013_fMRI_CPRO.txt', you would write '_fMRI_CPRO.txt' for this parameter
%   subjNums: This should be an array of the subject numbers of the EDATs you would like to import (e.g., [013, 014, 016, ...] etc)

%Defaults
% importFileEndString = '_fMRI_CPRO.txt';      %This requires that the filename start with the subject ID followed by a "_" symbol


%subjNums = [024];

EDATImport = cell(length(subjNums),1);

% tmp
% Example importFile
%importFile = '/projects/IndivRITL/data/rawdata/007/behavdata/fMRI_Behavioral/007_fMRI_CPRO.txt'
%importFile = '/projects/IndivRITL/data/rawdata/024/behavdata/fMRI_Behavioral/024_fMRI_CPRO.txt'


importNum=1;
for subjNum=subjNums
    dataName = 'fMRI_CPRO_IndivRITL';
    dataName = [num2str(subjNum) '_' dataName];
    % Adjsut for zeros
    if subjNum >= 100
        subjNum = num2str(subjNum);
    else
        subjNum = ['0' num2str(subjNum)];
    end
    importFile = ['/projects/IndivRITL/data/rawdata/' subjNum '/behavdata/fMRI_Behavioral/' subjNum importFileEndString];
    % importFile = [subjNum{1} importFileEndString];
    disp(['Importing file: ' importFile])
   
    subjID = subjNum;  %textscan(importFile, '%s', 'Delimiter', '_');
    % subjID = subjID{1}{1};
    
    disp(['Importing subject ' subjID])
    
    fid = fopen(importFile, 'r');
    importedLines = textscan(fid, '%s', 'Delimiter', '\n', 'TreatAsEmpty', 'NULL');
    importedLines = importedLines{1};
    % Remove top line, which is just the filename
    importedLines = importedLines(1:end);
    
    headerImport=textscan(importedLines{1}, '%s', 'Delimiter', '\t', 'TreatAsEmpty', 'NULL');
    edatImport = cell(length(importedLines),length(headerImport{1}));
    
    for lineNum = 1:length(importedLines)
        import=textscan(importedLines{lineNum}, '%s', 'Delimiter', '\t','TreatAsEmpty', 'NULL');
        edatImport(lineNum,:) = import{1}';
    end
    fclose(fid);
    
    EDATImport{importNum}.EDATData = edatImport;
    EDATImport{importNum}.SubjID = subjID;
    EDATImport{importNum}.DataName = dataName;
    
    importNum=importNum+1;
end
