% Taku Ito
% 08/31/2016

% This is a script that dilates individual parcels by 10mm
% original purpose was to use this as a mask to exclude any vertices within 10mm of a parcel from region-to-region ActFlow

parcelRange = 1:360;
dilateMM = 10;
basedir = ['/projects2/ModalityControl2/data/GlasserKKPartition/'];
outdir = [basedir 'ParcelLabels/'];

dlabelFile = [outdir 'Q1-Q6_RelatedParcellation210.LR.CorticalAreas_dil_Colors.32k_fs_LR.dlabel.nii'];
leftSurfaceFile = [outdir 'Q1-Q6_RelatedParcellation210.L.inflated_MSMAll_2_d41_WRN_DeDrift.32k_fs_LR.surf.gii'];
rightSurfaceFile = [outdir 'Q1-Q6_RelatedParcellation210.R.inflated_MSMAll_2_d41_WRN_DeDrift.32k_fs_LR.surf.gii'];

%% Dilate and write to file using wb_command -cifti-dilate
execute = 0;
if execute==1
    dlabelGifti = ciftiopen(dlabelFile,'wb_command');

    for parcel=parcelRange

        disp(['Dilating parcel ' num2str(parcel)])
        
        % Find vertices not part of this parcel
        ind = (dlabelGifti.cdata~=parcel);
        % Copy original gifti to new variable
        parcelGifti = dlabelGifti;
        % 0 out all other parcels to make a single ROI parcel
        parcelGifti.cdata(ind) = 0;

        % Specify output filename 
        outfile = [outdir 'Parcel' num2str(parcel) 'Mask.dlabel.nii'];
        ciftisave(parcelGifti, outfile, 'wb_command');

        % Now dilate the parcel mask with wb_command
        dilatedfile = [outdir 'Parcel' num2str(parcel) 'MaskDil10mm.dlabel.nii'];
        wbcommand = ['wb_command -cifti-dilate ' outfile ' COLUMN ' num2str(dilateMM) ' ' num2str(dilateMM) ' ' dilatedfile ' -left-surface ' leftSurfaceFile ' -right-surface ' rightSurfaceFile];

        unix(wbcommand);
    end

end


%% Read in all dilated (individual) parcels and then aggregate into a CSV (each column for a different parcel
execute = 1;
if execute==1
    
    parcelMat = zeros(64984,length(parcelRange));
    for parcel=parcelRange

        disp(['Loading in parcel file ' num2str(parcel)])
        % Specify input filename 
        infile = [outdir 'Parcel' num2str(parcel) 'MaskDil10mm.dlabel.nii'];

        parcelarray = ft_read_cifti(infile);

        % put new parcel location into a csv
        parcelMat(:,parcel) = parcelarray.x1;

    end

    % Write out parcel array into csv
    outfile = [outdir 'GlasserParcelsAll_Dilated.csv'];
    csvwrite(outfile, parcelMat);
end




