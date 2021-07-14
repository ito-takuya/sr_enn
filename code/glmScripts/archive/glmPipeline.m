% Taku Ito
% 1/13/2017

% GLM master pipeline for StroopActFlow
%subjNums = {'013','014','016','017','018','021','023','024','025','026','027','028','030','031','032','033','034','035','037','038','039','040','041','042','043','045','046','047','048','049','050','053','055','056','057','058','062','063','064','066','067','068','069','070','072','074','075','076','077','081','082','085','086','087','088','090','092','093','094','095','097','098','099','101','102','103','104','105','106','108','109','110','111','112','114','115','117','118','119','120','121','122','123','124','125','126','127','128','129','130','131','132','134','135','136','137','138','139','140','141'};

subjNums = {'013','014','016','017','018','021','023','024','026','027','028','030','031','032','033','034','035','037','038','039','040','041','042','043','045','046','047','048','049','050','053','055','056','057','058','062','063','066','067','068','069','070','072','074','075','076','077','081','085','086','087','088','090','092','093','094','095','097','098','099','101','102','103','104','105','106','108','109','110','111','112','114','115','117','119','120','121','122','123','124','125','126','127','128','129','130','131','132','134','135','136','137','138','139','140','141'};

%% Rest GLM on 64k Surface
execute = 0;
% Parameters
gsr = 0;
nproc = 8;
if execute == 1

    X = cell(length(subjNums),1);

    scount = 1;
    for subjNum=subjNums
        subj = subjNum{1};

        disp(['Running 64k Surface GLM on subject ' subj ' on ' num2str(nproc) ' processes'])
        X{scount} = surfaceGLM_rest(subj, gsr, nproc); 

    end
end

%% Rest GLM on Glasser Parcels
execute = 1;
% Parameters
gsr = 0;
nproc = 8;
if execute == 1

    X = cell(length(subjNums),1);

    scount = 1;
    for subjNum=subjNums
        subj = subjNum{1};

        disp(['Running Glasser Parcels Rest GLM on subject ' subj ' on ' num2str(nproc) ' processes'])
        X{scount} = GlasserGLM_restdata(subj, gsr, nproc); 

    end
end

%% Task GLM on 64k Surface -- Motor Response
execute = 0;
% Parameters
gsr = 1;
nproc = 10;
if execute == 1

    X = cell(length(subjNums),1);

    scount = 1;
    for subjNum=subjNums
        subj = subjNum{1};

        disp(['Running 64k Surface GLM on subject ' subj ' on ' num2str(nproc) ' processes'])
        X{scount} = SurfaceGLM_MotorResponse(subj, gsr, nproc); 

    end
end

%% Task GLM on 64k Surface -- Input stimulus GLMs
execute = 0;
% Parameters
gsr = 1;
nproc = 15;
if execute == 1

    %inputStimuli = {'Color', 'Ori', 'Constant', 'Pitch'};
    inputStimuli = {'Color'};

    for i=1:length(inputStimuli)
        inputtype = inputStimuli{i};

        scount = 1;
        for subjNum=subjNums
            subj = subjNum{1};

            disp(['Running 64k Surface GLM on subject ' subj ' on ' num2str(nproc) ' processes'])
            X = SurfaceGLM_InputStimulus(subj, gsr, inputtype, nproc); 

        end
    end
end

