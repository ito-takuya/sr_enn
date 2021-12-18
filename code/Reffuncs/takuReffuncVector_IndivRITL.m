function reffuncVector = takuReffuncVector_IndivRITL(reffunc)

    %Taku Ito
    %01/09/2015

    %Construct a reffunc vector array of all concatenated TRs by subject.  (i.e., first set of TRs are subject 1, 2nd set of TRs are subject 2, etc.)

    %Create empty refFunc Vector
    %for number of subjects * number of TRs for each subjectName

    numSubjs = length(reffunc);
    numTRs = length(reffunc{1}.RefFunc); %This assumes all subjects have the same number of TRs, which should be the case

    refFuncVector = cell(numTRs*numSubjs,1); 
    subjCounter=1;
    subjTRCounter=1;
    for totalTR=1:length(refFuncVector)
        reffuncVector{totalTR}=reffunc{subjCounter}.RefFunc{subjTRCounter};
        subjTRCounter=subjTRCounter+1;
        %set the counters to appropriate subject and subject's TR count
        if (subjTRCounter==(numTRs + 1))
            subjTRCounter=1;
            subjCounter=subjCounter+1;
        end
        
    end

end
