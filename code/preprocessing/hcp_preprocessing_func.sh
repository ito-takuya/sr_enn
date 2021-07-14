#!/bin/bash
# Taku Ito
# 10/29/14

# Cole Lab incorporation of HCP Pipelines for use in analysis of IndivRITL data set (RUBIC protocol)
# Will quit script if there is an error detected
#set -e
#########################
source opts.shlib # command line option functions
########################


show_usage() {
    cat <<EOF
    
This script serves as a function to run an individual subject through the HCP Pipeline, along with other necessary commands to fit their pipeline to the IndivRITL analysis.

hcp_preprocessing_func.sh

Usage: hcp_preprocessing_func.sh [options]

    --server=<servername>       (required) The name of the server you are running this from (i.e., "colelabmac", "colelablinux", "nm3")
    --subj=<subjectID>          (required) The subject number 
    --dirSetUp=<"true">         (optional) Input "true", if you want to set up directory structure for this subject
    --anatomicalRecon=<"true">  (optional) Input "true", if you want to run a DICOM reconstruction of the anatomicals
    --epiRecon=<"true">         (optional) Input "true", if you want to run a DICOM reconstruction of the EPIs
    --preFS=<"true">            (optional) Input "true", if you want to run the PreFS HCP Node
    --FS=<"true">               (optional) Input "true", if you want to run the FS HCP Node
    --postFS=<"true">           (optional) Input "true", if you want to run the Post FS HCP Node
    --fmriVol=<"true">          (optional) Input "true", if you want to run the fMRIVolume processing of the HCP Node
    --fmriSurf=<"true">         (optional) Input "true", if you want to run the fMRISurface processing of the HCP node
    --diff=<"true">             (optional) Input "true", if you want to run the Diffusion processing of the HCP node (NOTE: As of 1/29/15, this is still not functioning properly!)
    --createMasks=<"true">           (optional) Input "true", if you want to create whole brain, gray, white and ventricle masks for fMRI volume processed data (non-HCP node)
    --concatenateRuns=<"true">  (optional) Input "true", if you want to concatenate all task runs into analysis directory, titled Task_allruns.nii.gz
    --tsExtract=<"true">        (optional) Input "true", if you want to extract the time series for whole brain, white matter, ventricle time series of both Rest1.nii.gz and Task_allruns.nii.gz. Outputs timeseries into subject's analysis directory
EOF
    exit 1
}


opts_ShowVersionIfRequested $@

if opts_CheckForHelpRequest $@; then
    show_usage
fi


#Input Variables
server=`opts_GetOpt1 "--server" $@`
subj=`opts_GetOpt1 "--subj" $@`
dirSetUp=`opts_GetOpt1 "--dirSetUp" $@`
anatomicalRecon=`opts_GetOpt1 "--anatomicalRecon" $@`
epiRecon=`opts_GetOpt1 "--epiRecon" $@`
preFS=`opts_GetOpt1 "--preFS" $@`
FS=`opts_GetOpt1 "--FS" $@`
postFS=`opts_GetOpt1 "--postFS" $@`
fmriVol=`opts_GetOpt1 "--fmriVol" $@`
fmriSurf=`opts_GetOpt1 "--fmriSurf" $@`
diff=`opts_GetOpt1 "--diff" $@`
createMasks=`opts_GetOpt1 "--createMasks" $@`
concatenateRuns=`opts_GetOpt1 "--concatenateRuns" $@`
tsExtract=`opts_GetOpt1 "--tsExtract" $@`


#########################
# Set up HCP Environment - This is the customized environment for ColeLabMac Server
# Figure out which server is being used (mac or linux)
if [ -z "$server" ]; then
    echo "Missing required option. Indicate which server you're using! Exiting script..."
    exit
elif [ "$server" == "colelabmac" ]; then
    HCPPipe=/Applications/HCP_Pipelines/Pipelines-3.4.1
    EnvScript=${HCPPipe}/Examples/Scripts/SetUpHCPPipeline_Custom.sh
elif [ "$server" == "colelablinux" ]; then
    HCPPipe=/usr/local/HCP_Pipelines
    EnvScript=${HCPPipe}/Examples/Scripts/SetUpHCPPipeline.sh
elif [ "$server" == "nm3" ]; then
    HCPPipe=/home/ti61/Pipelines/
    EnvScript=${HCPPipe}/Examples/Scripts/SetUpHCPPipeline.sh
fi
# Set up HCP Pipeline Environment
. ${EnvScript}
########################

# First need to reconstruct data

#########################
# Set up Subject Directory Parameters
if [ "$server" == "nm3" ]; then
    basedir="/scratch/ti61/projects/IndivRITL"
else
    basedir="/projects/IndivRITL"
fi
datadir="${basedir}/data"
subjdir=${datadir}/${subj}
if [ ! -e $subjdir ]; then mkdir $subjdir; fi 
unprocesseddir="${datadir}/${subj}/unprocessed"
if [ ! -e $unprocesseddir ]; then mkdir $unprocesseddir; fi 
rawdatadir="${datadir}/rawdata/${subj}"
if [ ! -e $rawdatadir ]; then mkdir $rawdatadir; fi 
analysisdir=${subjdir}/analysis
if [ ! -e $analysisdir ]; then mkdir $analysisdir; fi
subjmaskdir=${subjdir}/masks
if [ ! -e $subjmaskdir ]; then mkdir $subjmaskdir; fi 

#########################


#########################
### SPECIFY SCAN LISTS ###
# Anatomicals
T1w=(`ls ${rawdatadir}/ | grep t1_mpr_ns_sag_p2_iso_32Channel_[0-9] || true;` )
T2w=(`ls ${rawdatadir}/ | grep t2_spc_sag_p2_iso_[0-9] || true;` )
# Spin Echo Maps/Gradient Field Maps
SE_AP=(`ls ${rawdatadir}/ | grep cmrr_mbep2d_se_AP_[0-9] || true;` ) # AP is the NEGATIVE PHASE encoding (think a 
SE_PA=(`ls ${rawdatadir}/ | grep cmrr_mbep2d_se_PA_[0-9] || true;` ) # PA is the POSITIVE PHASE encoding
GRE_FIELDMAP=(`ls ${rawdatadir}/ | grep gre_field_mapping_[0-9] || true;` )
# Multiband EPI sequences 
# Rest scans
RestEPIs=(`ls ${rawdatadir}/ | grep mbep2d_bold_2mm_MB8_Rest_[0-9] || true;` ) # This returns an array of the scans delimited by spaces
# SBRefs for Rest scans
RestEPIs_SBRef=(`ls ${rawdatadir}/ | grep mbep2d_bold_2mm_MB8_Rest_SBRef_[0-9] || true;` )
# Task scans
TaskEPIs=(`ls ${rawdatadir}/ | grep mbep2d_bold_2mm_MB8_Task_8x_[0-9] || true;` )
# SBRefs for task scans
TaskEPIs_SBRef=(`ls ${rawdatadir}/ | grep mbep2d_bold_2mm_MB8_Task_8x_SBRef_[0-9] || true;` )
# Task localizer
TaskLoc=(`ls ${rawdatadir}/ | grep mbep2d_bold_2mm_MB8_TaskLoc_[0-9] || true;` )
# SBRef for task localizer
TaskLoc_SBRef=(`ls ${rawdatadir}/ | grep mbep2d_bold_2mm_MB8_TaskLoc_SBRef_[0-9] || true;` )


# Diffusion
#*** Don't worry for now ***#
DIFF="${rawdatadir}/cmrr_mbep2d_diff_2mm_MB3_[0-9]"
DIFF_SB="${rawdatadir}/cmrr_mbep2d_diff_2mm_MB3_SBRef_[0-9]"
#########################

#########################
# HCP Conventions and Parameters - shouldn't need to edit this
#PostFreeSurfer input file names
#Input Variables
SurfaceAtlasDIR="${HCPPIPEDIR_Templates}/standard_mesh_atlases"
GrayordinatesSpaceDIR="${HCPPIPEDIR_Templates}/91282_Greyordinates"
GrayordinatesResolutions="2" #Usually 2mm, if multiple delimit with @, must already exist in templates dir
HighResMesh="164" #Usually 164k vertices
LowResMeshes="32" #Usually 32k vertices, if multiple delimit with @, must already exist in templates dir
SubcorticalGrayLabels="${HCPPIPEDIR_Config}/FreeSurferSubcorticalLabelTableLut.txt"
FreeSurferLabels="${HCPPIPEDIR_Config}/FreeSurferAllLut.txt"
ReferenceMyelinMaps="${HCPPIPEDIR_Templates}/standard_mesh_atlases/Conte69.MyelinMap_BC.164k_fs_LR.dscalar.nii"
# RegName="MSMSulc" #MSMSulc is recommended, if binary is not available use FS (FreeSurfer)
RegName="FS" 
#########################

#########################
# Data and scan parameters
SmoothingFWHM="2"
DwellTime_SE="0.00069" # the dwell time or echo spacing of the SE Maps (see protocol)
DwellTime_fMRI="0.00069" # the dwell time or echo spacing of the fMRI multiband sequence (see protocol)
T1wSampleSpacing="0.0000074" # This parameter can be found at DICOM field (0019,1028) (use command `dicom_hdr *001.dcm | grep "0019 1018"`
T2wSampleSpacing="0.0000021" # This parameter can be found at DICOM field (0019,1028) (use command `dicom_hdr *001.dcm | grep "0019 1018"`
# Default parameters (should not need to be changed for this study
fmrires="2"
brainsize="150"
seunwarpdir="y" # AP/PA is Y
unwarpdir="-y" # unwarp direction for A >> P phase encoded data is -y (think about how anterior to posterior coordinate direction is -y). It follows that unwarpdir for P >> A collected data would be "y", but we have not collected this data for the IndivRITL study.
numTRsPerTaskRun=581


# Anatomical templates for this study (MNI templates)
t1template="${HCPPipe}/global/templates/MNI152_T1_0.8mm.nii.gz"
t1template2mm="${HCPPipe}/global/templates/MNI152_T1_2mm.nii.gz"
t1templatebrain="${HCPPipe}/global/templates/MNI152_T1_0.8mm_brain.nii.gz"
t2template="${HCPPipe}/global/templates/MNI152_T2_0.8mm.nii.gz"
t2templatebrain="${HCPPipe}/global/templates/MNI152_T2_0.8mm_brain.nii.gz"
t2template2mm="${HCPPipe}/global/templates/MNI152_T2_2mm.nii.gz"
templatemask="${HCPPipe}/global/templates/MNI152_T1_0.8mm_brain_mask.nii.gz"
template2mmmask="${HCPPipe}/global/templates/MNI152_T1_2mm_brain_mask_dil.nii.gz"
#########################

#########################
# Set up directories for behavioral data
# This is not part of the HCP pipeline
if [ -z "$dirSetUp" ]; then
    echo "Skipping directory set up"
elif [ "$dirSetUp" == "true" ]; then
    echo "Setting up directories"
    behavdir=${rawdatadir}/behavdata
    duncan=${behavdir}/Duncan
    mobile=${behavdir}/MobileSurveys
    nih=${behavdir}/NIHToolBox
    opspan=${behavdir}/OpSpan
    fmri=${behavdir}/fMRI_Behavioral
    cproprac=${behavdir}/CPRO_Practice
    if [ ! -e $behavdir ]; then mkdir $behavdir; fi 
    if [ ! -e $duncan ]; then mkdir $duncan; fi 
    if [ ! -e $mobile ]; then mkdir $mobile; fi 
    if [ ! -e $nih ]; then mkdir $nih; fi 
    if [ ! -e $opspan ]; then mkdir $opspan; fi 
    if [ ! -e $fmri ]; then mkdir $fmri; fi 
    if [ ! -e $cproprac ]; then mkdir $cproprac; fi
    # Allow other users without admin privileges to edit these directories
    chmod -R 775 ${rawdatadir}
fi
#########################
# DICOM to NIFTI reconstruction using Freesurfer (Everything is reconstructed EXCEPT Diffusion data)
# This is NOT part of the HCP pipeline
# Reconstruct Anatomicals and Field Maps
if [ -z "$anatomicalRecon" ]; then
    echo "Skipping anatomical reconstruction"
elif [ $anatomicalRecon = true ]; then
    

    if [ ! -e ${unprocesseddir}/T1w ]; then mkdir ${unprocesseddir}/T1w; fi 
    if [ ! -e ${unprocesseddir}/T2w ]; then mkdir ${unprocesseddir}/T2w; fi 
    if [ ! -e ${unprocesseddir}/SE_Maps ]; then mkdir ${unprocesseddir}/SE_Maps; fi 
    if [ ! -e ${unprocesseddir}/GRE_FieldMap ]; then mkdir ${unprocesseddir}/GRE_FieldMap; fi 
    
    echo "Reconstructing T1 image"
    mri_convert ${rawdatadir}/${T1w}/*0001.dcm --in_type siemens --out_type nii ${unprocesseddir}/T1w/T1w.nii.gz
    echo "Reconstructing T2 image"
    mri_convert ${rawdatadir}/${T2w}/*0001.dcm --in_type siemens --out_type nii ${unprocesseddir}/T2w/T2w.nii.gz
    echo "Reconstructing Spin Echo Field Maps"
    mri_convert ${rawdatadir}/${SE_AP}/*0001.dcm --in_type siemens --out_type nii ${unprocesseddir}/SE_Maps/SE_AP.nii.gz
    mri_convert ${rawdatadir}/${SE_PA}/*0001.dcm --in_type siemens --out_type nii ${unprocesseddir}/SE_Maps/SE_PA.nii.gz
    echo "Reconstructing Gradient Field Map"
    mri_convert ${rawdatadir}/${GRE_FIELDMAP}/*0001.dcm --in_type siemens --out_type nii ${unprocesseddir}/GRE_FieldMap/GRE_FieldMap.nii.gz
    echo "Resampling Field Map images to RPI coordinates"
    3dresample -overwrite -orient RPI -inset ${unprocesseddir}/SE_Maps/SE_AP.nii.gz -prefix ${unprocesseddir}/SE_Maps/SE_AP.nii.gz
    3dresample -overwrite -orient RPI -inset ${unprocesseddir}/SE_Maps/SE_PA.nii.gz -prefix ${unprocesseddir}/SE_Maps/SE_PA.nii.gz
    3dresample -overwrite -orient RPI -inset ${unprocesseddir}/GRE_FieldMap/GRE_FieldMap.nii.gz -prefix ${unprocesseddir}/GRE_FieldMap/GRE_FieldMap.nii.gz


fi 
# Reconstruct EPI scans (task and rest)
if [ -z "$epiRecon" ]; then
    echo "Skipping EPI reconstruction"
elif [ $epiRecon = true ]; then
    echo "Reconstructing Rest EPI Scans..."
    # make directories for rest scan
    for ((i=0;i<${#RestEPIs[@]};++i)); do
        # Count the scans starting from 1 (as opposed to 0)
        let count=${i}+1

        # create the unprocessed directory
        if [ ! -e ${unprocesseddir}/Rest${count} ]; then mkdir ${unprocesseddir}/Rest${count}; fi 
        echo "Converting Scan ${i} for Rest scan"
        # reconstruct using mri_convert (freesurfer) for both the epi scan and the sbref
        mri_convert ${rawdatadir}/${RestEPIs[i]}/*0001.dcm --in_type siemens --out_type nii ${unprocesseddir}/Rest${count}/Rest${count}.nii.gz
        mri_convert ${rawdatadir}/${RestEPIs_SBRef[i]}/*0001.dcm --in_type siemens --out_type nii ${unprocesseddir}/Rest${count}/Rest${count}_SBRef.nii.gz
        
        # resample the data to RPI coordinates
        echo "Resampling Rest images to RPI coordinates"
        3dresample -overwrite -orient RPI -inset ${unprocesseddir}/Rest${count}/Rest${count}.nii.gz -prefix ${unprocesseddir}/Rest${count}/Rest${count}.nii.gz
        3dresample -overwrite -orient RPI -inset ${unprocesseddir}/Rest${count}/Rest${count}_SBRef.nii.gz -prefix ${unprocesseddir}/Rest${count}/Rest${count}_SBRef.nii.gz
    done

    echo "Reconstructing Task EPI Scans..."
    # make directories for task scan
    for ((i=0;i<${#TaskEPIs[@]};++i)); do
        # Count the scans starting from 1 (as opposed to 0). This is for the numbering for the nifti filename.
        let count=${i}+1

        # create the unprocessed directory
        if [ ! -e ${unprocesseddir}/Task${count} ]; then mkdir ${unprocesseddir}/Task${count}; fi 
        echo "Converting Scan ${i} for Task scan"
        # reconstruct using mri_convert (freesurfer) for both the epi scan and the sbref
        mri_convert ${rawdatadir}/${TaskEPIs[i]}/*0001.dcm --in_type siemens --out_type nii ${unprocesseddir}/Task${count}/Task${count}.nii.gz
        mri_convert ${rawdatadir}/${TaskEPIs_SBRef[i]}/*0001.dcm --in_type siemens --out_type nii ${unprocesseddir}/Task${count}/Task${count}_SBRef.nii.gz
        
        # resample the data to RPI coordinates
        echo "Resampling Task images to RPI coordinates"
        3dresample -overwrite -orient RPI -inset ${unprocesseddir}/Task${count}/Task${count}.nii.gz -prefix ${unprocesseddir}/Task${count}/Task${count}.nii.gz
        3dresample -overwrite -orient RPI -inset ${unprocesseddir}/Task${count}/Task${count}_SBRef.nii.gz -prefix ${unprocesseddir}/Task${count}/Task${count}_SBRef.nii.gz
    done

    echo "Reconstructing Task Localizer"
    if [ ! -e ${unprocesseddir}/TaskLoc/ ]; then mkdir ${unprocesseddir}/TaskLoc; fi
    mri_convert ${rawdatadir}/${TaskLoc}/*0001.dcm --in_type siemens --out_type nii ${unprocesseddir}/TaskLoc/TaskLoc.nii.gz
    mri_convert ${rawdatadir}/${TaskLoc_SBRef}/*0001.dcm --in_type siemens --out_type nii ${unprocesseddir}/TaskLoc/TaskLoc_SBRef.nii.gz
    echo "Resampling Task Localizer images to RPI coordinates"
    3dresample -overwrite -orient RPI -inset ${unprocesseddir}/TaskLoc/TaskLoc.nii.gz -prefix ${unprocesseddir}/TaskLoc/TaskLoc.nii.gz
    3dresample -overwrite -orient RPI -inset ${unprocesseddir}/TaskLoc/TaskLoc_SBRef.nii.gz -prefix ${unprocesseddir}/TaskLoc/TaskLoc_SBRef.nii.gz



fi
#########################


#########################
# Start first node of HCP Pipeline: PreFreeSurferPipeline
if [ -z "$preFS" ]; then
    echo "Skipping PreFS HCP Node"
elif [ $preFS = true ]; then

    ${HCPPipe}/PreFreeSurfer/PreFreeSurferPipeline.sh \
        --path="${datadir}" \
        --subject="${subj}" \
        --t1="${unprocesseddir}/T1w/T1w.nii.gz" \
        --t2="${unprocesseddir}/T2w/T2w.nii.gz" \
        --t1template="${t1template}" \
        --t1templatebrain="${t1templatebrain}" \
        --t1template2mm="${t1template2mm}" \
        --t2template="${t2template}" \
        --t2templatebrain="$t2templatebrain" \
        --t2template2mm="$t2template2mm" \
        --templatemask="$templatemask" \
        --template2mmmask="$template2mmmask" \
        --brainsize="${brainsize}" \
        --fmapmag="NONE" \
        --fnirtconfig="${HCPPipe}/global/config/T1_2_MNI152_2mm.cnf" \
        --SEPhaseNeg="${unprocesseddir}/SE_Maps/SE_AP.nii.gz" \
        --SEPhasePos="${unprocesseddir}/SE_Maps/SE_PA.nii.gz" \
        --echospacing="$DwellTime_SE" \
        --seunwarpdir="${seunwarpdir}" \
        --t1samplespacing="$T1wSampleSpacing" \
        --t2samplespacing="$T2wSampleSpacing" \
        --unwarpdir="z" \
        --grdcoeffs="NONE" \
        --avgrdcmethod="TOPUP" \
        --topupconfig="${HCPPIPEDIR_Config}/b02b0.cnf" \
        --printcom=""
fi
#########################

#########################
# Start second node of HCP Pipeline: FreeSurferPipeline
if [ -z "$FS" ]; then
    echo "Skipping Freesurfer HCP node"
elif [ $FS = true ]; then
    # limit number of threads of FS
    export OMP_NUM_THREADS=3
    ${HCPPipe}/FreeSurfer/FreeSurferPipeline.sh \
        --subject="${subj}" \
        --subjectDIR="${datadir}/${subj}/T1w" \
        --t1="${datadir}/${subj}/T1w/T1w_acpc_dc_restore.nii.gz" \
        --t1brain="${datadir}/${subj}/T1w/T1w_acpc_dc_restore_brain.nii.gz" \
        --t2="${datadir}/${subj}/T1w/T2w_acpc_dc_restore.nii.gz" 
fi

#########################
# Start third node of HCP Pipeline: PostFreeSurferPipeline
if [ -z "$postFS" ]; then
    echo "Skipping PostFS HCP node"
elif [ $postFS = true ]; then

    ${HCPPipe}/PostFreeSurfer/PostFreeSurferPipeline.sh \
        --path="${datadir}" \
        --subject="${subj}" \
        --surfatlasdir="$SurfaceAtlasDIR" \
        --grayordinatesdir="$GrayordinatesSpaceDIR" \
        --grayordinatesres="$GrayordinatesResolutions" \
        --hiresmesh="$HighResMesh" \
        --lowresmesh="$LowResMeshes" \
        --subcortgraylabels="$SubcorticalGrayLabels" \
        --freesurferlabels="$FreeSurferLabels" \
        --refmyelinmaps="$ReferenceMyelinMaps" \
        --regname="$RegName" \
        --printcom=""

fi
#########################


#########################
# Start fourth node of HCP Pipeline: GenericfMRIVolumeProcessing
if [ -z "$fmriVol" ]; then
    echo "Skipping fMRIVolumeProcessing node"
elif [ $fmriVol = true ]; then
    # need to iterate through each rest scan, and then each task scan

    ## Rest Scan(s)
    for ((i=1;i<=${#RestEPIs[@]};++i)); do
        # Count the scans starting from 1 (as opposed to 0). This is for the numbering for the nifti filename.
        echo "Running fMRI Volume processing on Rest scan #${i}" 
        fmriname="Rest${i}"
        fmritcs="${unprocesseddir}/Rest${i}/Rest${i}.nii.gz"
        fmriscout="${unprocesseddir}/Rest${i}/Rest${i}_SBRef.nii.gz"
        ${HCPPipe}/fMRIVolume/GenericfMRIVolumeProcessingPipeline.sh \
            --path="${datadir}" \
            --subject="${subj}" \
            --fmriname="${fmriname}" \
            --fmritcs="${fmritcs}" \
            --fmriscout="${fmriscout}" \
            --SEPhaseNeg="${unprocesseddir}/SE_Maps/SE_AP.nii.gz" \
            --SEPhasePos="${unprocesseddir}/SE_Maps/SE_PA.nii.gz" \
            --fmapmag="NONE" \
            --fmapphase="NONE" \
            --echospacing="$DwellTime_fMRI" \
            --echodiff="NONE" \
            --unwarpdir="${unwarpdir}" \
            --fmrires="$fmrires" \
            --dcmethod="TOPUP" \
            --gdcoeffs="NONE" \
            --printcom="" \
            --topupconfig="${HCPPIPEDIR_Config}/b02b0.cnf"
    done

    ## Task Scans
    for ((i=1;i<=${#TaskEPIs[@]};++i)); do
        # Count the scans starting from 1 (as opposed to 0). This is for the numbering for the nifti filename.
        echo "Running fMRI Volume processing on Task scan #${i}" 
        
        fmriname="Task${i}"
        fmritcs="${unprocesseddir}/Task${i}/Task${i}.nii.gz"
        fmriscout="${unprocesseddir}/Task${i}/Task${i}_SBRef.nii.gz"
        ${HCPPipe}/fMRIVolume/GenericfMRIVolumeProcessingPipeline.sh \
            --path="${datadir}" \
            --subject="${subj}" \
            --fmriname="${fmriname}" \
            --fmritcs="${fmritcs}" \
            --fmriscout="${fmriscout}" \
            --SEPhaseNeg="${unprocesseddir}/SE_Maps/SE_AP.nii.gz" \
            --SEPhasePos="${unprocesseddir}/SE_Maps/SE_PA.nii.gz" \
            --fmapmag="NONE" \
            --fmapphase="NONE" \
            --echospacing="$DwellTime_fMRI" \
            --echodiff="NONE" \
            --unwarpdir="${unwarpdir}" \
            --fmrires="$fmrires" \
            --dcmethod="TOPUP" \
            --gdcoeffs="NONE" \
            --printcom="" \
            --topupconfig="${HCPPIPEDIR_Config}/b02b0.cnf" 
    done
   
        
    echo "Running fMRI Volume processing on Task Localizer scan" 
    
    fmriname="TaskLoc"
    fmritcs="${unprocesseddir}/TaskLoc/TaskLoc.nii.gz"
    fmriscout="${unprocesseddir}/TaskLoc/TaskLoc_SBRef.nii.gz"
    ${HCPPipe}/fMRIVolume/GenericfMRIVolumeProcessingPipeline.sh \
        --path="${datadir}" \
        --subject="${subj}" \
        --fmriname="${fmriname}" \
        --fmritcs="${fmritcs}" \
        --fmriscout="${fmriscout}" \
        --SEPhaseNeg="${unprocesseddir}/SE_Maps/SE_AP.nii.gz" \
        --SEPhasePos="${unprocesseddir}/SE_Maps/SE_PA.nii.gz" \
        --fmapmag="NONE" \
        --fmapphase="NONE" \
        --echospacing="$DwellTime_fMRI" \
        --echodiff="NONE" \
        --unwarpdir="${unwarpdir}" \
        --fmrires="$fmrires" \
        --dcmethod="TOPUP" \
        --gdcoeffs="NONE" \
        --printcom="" \
        --topupconfig="${HCPPIPEDIR_Config}/b02b0.cnf" 
fi
#########################

#########################
# Start fifth node of HCP Pipeline: GenericfMRISurfaceProcessing
if [ -z "$fmriSurf" ]; then
    echo "Skipping fMRI Surface Processing node"
elif [ $fmriSurf = true ]; then

    # need to iterate through each rest scan, and then each task scan

    ## Rest Scan(s)
    for ((i=1;i<=${#RestEPIs[@]};++i)); do
        # Count the scans starting from 1 (as opposed to 0). This is for the numbering for the nifti filename.
        
        # set input name
        fmriname="Rest${i}"

        ${HCPPipe}/fMRISurface/GenericfMRISurfaceProcessingPipeline.sh \
            --path="${datadir}" \
            --subject="${subj}" \
            --fmriname="${fmriname}" \
            --fmrires="$fmrires" \
            --lowresmesh="$LowResMeshes" \
            --smoothingFWHM=$SmoothingFWHM \
            --grayordinatesres="$GrayordinatesResolutions" \
            --regname=$RegName
    done

    ## Task Scan(s)
    for ((i=1;i<=${#TaskEPIs[@]};++i)); do
        # Count the scans starting from 1 (as opposed to 0). This is for the numbering for the nifti filename.
        
        # set input name
        fmriname="Task${i}"

        ${HCPPipe}/fMRISurface/GenericfMRISurfaceProcessingPipeline.sh \
            --path="${datadir}" \
            --subject="${subj}" \
            --fmriname="${fmriname}" \
            --fmrires="$fmrires" \
            --lowresmesh="$LowResMeshes" \
            --smoothingFWHM=$SmoothingFWHM \
            --grayordinatesres="$GrayordinatesResolutions" \
            --regname=$RegName
    done
        
    
    # set input name
    fmriname="TaskLoc"

    ${HCPPipe}/fMRISurface/GenericfMRISurfaceProcessingPipeline.sh \
        --path="${datadir}" \
        --subject="${subj}" \
        --fmriname="${fmriname}" \
        --fmrires="$fmrires" \
        --lowresmesh="$LowResMeshes" \
        --smoothingFWHM=$SmoothingFWHM \
        --grayordinatesres="$GrayordinatesResolutions" \
        --regname=$RegName
fi
#########################



######################### NOT COMPLETED!
# Start 6th node of HCP Pipline: Diffusion
if [ -z "$diff" ]; then
    echo "Skipping Diffusion HCP node"
elif [ $diff = true ]; then

    if [ ! -e ${unprocesseddir}/Diffusion ]; then mkdir ${unprocesseddir}/Diffusion; fi 

    echo "Reconstructing DIFF and DIFF_SB"
    dcm2nii -g -o="${unprocesseddir}/Diffusion/" ${DIFF}/*.dcm
    dcm2nii -g -o="${unprocesseddir}/Diffusion/" ${DIFF_SB}/*.dcm

    
 #   ${HCPPIPEDIR}/DiffusionPreprocessing/DiffPreprocPipeline.sh \
 #     --posData="${PosData}" --negData="${NegData}" \
 #     --path="${StudyFolder}" --subject="${SubjectID}" \
 #     --echospacing="${EchoSpacing}" --PEdir=${PEdir} \
 #     --gdcoeffs="${Gdcoeffs}" \
 #     --printcom=$PRINTCOM



fi



#########################
# Create masks
if [ -z "$createMasks" ]; then
    echo "Skipping mask creation"
elif [ "$createMasks" == "true" ]; then
    
    echo "Creating gray, white, ventricle, whole brain masks for subject ${subj}..."

    # HCP standard to parcel out white v gray v ventricle matter
    segparc=${subjdir}/MNINonLinear/wmparc.nii.gz

    # Change to subjmaskdir
    pushd $subjmaskdir

    
    
    ###############################
    ### Create whole brain masks
    echo "Creating whole brain mask for subject ${subj}..."
    3dcalc -overwrite -a $segparc -expr 'ispositive(a)' -prefix ${subj}_wholebrainmask.nii.gz
    # Resample to functional space
    3dresample -overwrite -master ${subjdir}/MNINonLinear/Results/Rest1/Rest1.nii.gz -inset ${subj}_wholebrainmask.nii.gz -prefix ${subj}_wholebrainmask_func.nii.gz
    # Dilate mask by 1 functional voxel (just in case the resampled anatomical mask is off by a bit)
    3dLocalstat -overwrite -nbhd 'SPHERE(-1)' -stat 'max' -prefix ${subj}_wholebrainmask_func_dil1vox.nii.gz ${subj}_wholebrainmask_func.nii.gz
    
   

    ###############################
    ### Create gray matter masks
    echo "Creating gray matter masks for subject ${subj}..." 
    # Indicate the mask value set for wmparc.nii.gz
    # Gray matter mask set
    maskValSet="8 9 10 11 12 13 16 17 18 19 20 26 27 28 47 48 49 50 51 52 53 54 55 56 58 59 60 96 97 1000 1001 1002 1003 1004 1005 1006 1007 1008 1009 1010 1011 1012 1013 1014 1015 1016 1017 1018 1019 1020 1021 1022 1023 1024 1025 1026 1027 1028 1029 1030 1031 1032 1033 1034 1035 2000 2001 2002 2003 2004 2005 2006 2007 2008 2009 2010 2011 2012 2013 2014 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025 2026 2027 2028 2029 2030 2031 2032 2033 2034 2035"

    # Add segments to mask
    maskNum=1
    for maskval in $maskValSet
    do
	if [ ${maskNum} = 1 ]; then
            3dcalc -a $segparc -expr "equals(a,${maskval})" -prefix ${subj}mask_temp.nii.gz -overwrite
        else
            3dcalc -a $segparc -b ${subj}mask_temp.nii.gz -expr "equals(a,${maskval})+b" -prefix ${subj}mask_temp.nii.gz -overwrite
        fi
	let maskNum++
    done
    #Make mask binary
    3dcalc -a ${subj}mask_temp.nii.gz -expr 'ispositive(a)' -prefix ${subj}_gmMask.nii.gz -overwrite
    #Resample to functional space
    3dresample -overwrite -master ${subjdir}/MNINonLinear/Results/Rest1/Rest1.nii.gz -inset ${subj}_gmMask.nii.gz -prefix ${subj}_gmMask_func.nii.gz
    #Dilate mask by 1 functional voxel (just in case the resampled anatomical mask is off by a bit)
    3dLocalstat -overwrite -nbhd 'SPHERE(-1)' -stat 'max' -prefix ${subj}_gmMask_func_dil1vox.nii.gz ${subj}_gmMask_func.nii.gz
    
    rm -f ${subj}mask_temp.nii.gz
       
   
    
    ###############################
    ### Create white matter masks
    echo "Creating white matter masks for subject ${subj}..."

    # Indicate the mask value set for wmparc.nii.gz
    # White matter mask set
    maskValSet="250 251 252 253 254 255 3000 3001 3002 3003 3004 3005 3006 3007 3008 3009 3010 3011 3012 3013 3014 3015 3016 3017 3018 3019 3020 3021 3022 3023 3024 3025 3026 3027 3028 3029 3030 3031 3032 3033 3034 3035 4000 4001 4002 4003 4004 4005 4006 4007 4008 4009 4010 4011 4012 4013 4014 4015 4016 4017 4018 4019 4020 4021 4022 4023 4024 4025 4026 4027 4028 4029 4030 4031 4032 4033 4034 4035 5001 5002"

    # Add segments to mask
    maskNum=1
    for maskval in $maskValSet
    do
	if [ ${maskNum} = 1 ]; then
            3dcalc -a $segparc -expr "equals(a,${maskval})" -prefix ${subj}mask_temp.nii.gz -overwrite
        else
            3dcalc -a $segparc -b ${subj}mask_temp.nii.gz -expr "equals(a,${maskval})+b" -prefix ${subj}mask_temp.nii.gz -overwrite
        fi
	let maskNum++
    done
    #Make mask binary
    3dcalc -a ${subj}mask_temp.nii.gz -expr 'ispositive(a)' -prefix ${subj}_wmMask.nii.gz -overwrite
    #Resample to functional space
    3dresample -overwrite -master ${subjdir}/MNINonLinear/Results/Rest1/Rest1.nii.gz -inset ${subj}_wmMask.nii.gz -prefix ${subj}_wmMask_func.nii.gz
    #Subtract graymatter mask from white matter mask (avoiding negative #s)
    3dcalc -a ${subj}_wmMask_func.nii.gz -b ${subj}_gmMask_func_dil1vox.nii.gz -expr 'step(a-b)' -prefix ${subj}_wmMask_func_eroded.nii.gz -overwrite
    
    rm -f ${subj}mask_temp.nii.gz
          

    
    ###############################
    ### Create ventricle masks
    echo "Creating ventricle matter masks for subject ${subj}..."

    # Indicate the mask value set for wmparc.nii.gz
    # Ventricle mask set
    maskValSet="4 43 14 15"

    # Add segments to mask
    maskNum=1
    for maskval in $maskValSet
    do
	if [ ${maskNum} = 1 ]; then
            3dcalc -a $segparc -expr "equals(a,${maskval})" -prefix ${subj}mask_temp.nii.gz -overwrite
        else
            3dcalc -a $segparc -b ${subj}mask_temp.nii.gz -expr "equals(a,${maskval})+b" -prefix ${subj}mask_temp.nii.gz -overwrite
        fi
	let maskNum++
    done
    #Make mask binary
    3dcalc -a ${subj}mask_temp.nii.gz -expr 'ispositive(a)' -prefix ${subj}_ventricles.nii.gz -overwrite
    #Resample to functional space
    3dresample -overwrite -master ${subjdir}/MNINonLinear/Results/Rest1/Rest1.nii.gz -inset ${subj}_ventricles.nii.gz -prefix ${subj}_ventricles_func.nii.gz
    #Subtract graymatter mask from ventricles (avoiding negative #s)
    3dcalc -a ${subj}_ventricles_func.nii.gz -b ${subj}_gmMask_func_dil1vox.nii.gz -expr 'step(a-b)' -prefix ${subj}_ventricles_func_eroded.nii.gz -overwrite
    rm -f ${subjNum}mask_temp.nii.gz
    
    rm -f ${subj}mask_temp.nii.gz
          
    popd

fi 



#########################
# Concatenate Task Runs
if [ -z "$concatenateRuns" ]; then
    echo "Skipping run concatenation"
elif [ "$concatenateRuns" == "true" ]; then
    
    pushd ${analysisdir}

    echo "Concatenating the 8 task runs for subject ${subj}..."
    runs="1 2 3 4 5 6 7 8" 
    runList=""
    for run in $runs
    do
        runList="${runList} ${subjdir}/MNINonLinear/Results/Task${run}/Task${run}.nii.gz"
    done

    3dTcat -prefix Task_allruns.nii.gz ${runList}

    popd
fi
  
#########################
# Extract Timeseries
if [ -z "$tsExtract" ]; then
    echo "Skipping Timeseries Extraction"
elif [ "$tsExtract" == "true" ]; then

    pushd ${analysisdir}

    echo "Extracting timeseries from Rest1.nii.gz and Task_allruns.nii.gz using white matter masks, ventricles, and whole brain masks for subject ${subj}..."
    rest=${subjdir}/MNINonLinear/Results/Rest1/Rest1.nii.gz
    task=${analysisdir}/Task_allruns.nii.gz
    
    # Extract Rest TS using wm and ventricle masks
    3dmaskave -quiet -mask ${subjmaskdir}/${subj}_wmMask_func_eroded.nii.gz ${rest} > ${subj}_WM_timeseries_rest.1D
    3dmaskave -quiet -mask ${subjmaskdir}/${subj}_ventricles_func_eroded.nii.gz ${rest} > ${subj}_ventricles_timeseries_rest.1D
    3dmaskave -quiet -mask ${subjmaskdir}/${subj}_wholebrainmask_func_dil1vox.nii.gz ${rest} > ${subj}_wholebrainsignal_timeseries_rest.1D

    # Extract Task TS using wm and ventricle masks
    3dmaskave -quiet -mask ${subjmaskdir}/${subj}_wmMask_func_eroded.nii.gz ${task} > ${subj}_WM_timeseries_task.1D
    3dmaskave -quiet -mask ${subjmaskdir}/${subj}_ventricles_func_eroded.nii.gz ${task} > ${subj}_ventricles_timeseries_task.1D
    3dmaskave -quiet -mask ${subjmaskdir}/${subj}_wholebrainmask_func_dil1vox.nii.gz ${task} > ${subj}_wholebrainsignal_timeseries_task.1D
    
    popd
fi
    
    

######################### DOES THIS NEED TO BE DONE? OR DOES HCP ALREADY COVER THIS?
## Resting-state Motion Correction
#if [ -z "$restMotionCorrectoin" ]; then
#    echo "Skipping Timeseries Extraction"
#elif [ "$tsExtract" == "true" ]; then
#
#fi
#
#
#########################
# Task-based analysis!
# Start Run concatenation and GLM analysis
execute=0
if [ $execute = true ]; then
    # First need to concat all 8 Task Runs together
    # Create run list
    runList=""
    concatString="1D:"
    TRCount=0
    for ((i=1;i<=${#TaskEPIs[@]};++i)); do
        runList="${runList} ${subjdir}/MNINonLinear/Results/Task${i}/Task${i}.nii.gz"
        concatString="${concatString} ${TRCount}"
        TRCount=$(expr $TRCount + $numTRsPerTaskRun)
    done

    echo "-Concatenating task runs-"
    echo "Run list: ${runList}"
    echo "Concatenation string (onset times of each run): ${concatString}"
    # create Analysis directory
    if [ ! -e ${subjdir}/Analysis ]; then mkdir ${subjdir}/Analysis; fi 
    #3dTcat -prefix ${subjdir}/Analysis/Task_allruns ${runList}

    echo "-Running GLM-"
   
    # First resample freesurfer brain mask to functional space (From HCP Pipelines)
    3dresample -master ${subjdir}/Analysis/Task_allruns+tlrc -prefix ${subjdir}/Analysis/brainmask_fs_func.nii.gz -inset ${subjdir}/MNINonLinear/brainmask_fs.nii.gz -overwrite

    3dDeconvolve \
        -input ${subjdir}/Analysis/Task_allruns+tlrc \
        -concat "$concatString" \
        -mask ${subjdir}/Analysis/brainmask_fs_func.nii.gz \
        -polort A \
        -num_stimts 1 \
        -stim_times 1 ${subjdir}/timingfiles/stime_013_stimfile_IndivRITL_PilotGLM_EV1_task.1D.01.1D 'BLOCK(1,1)' -stim_label 1 Task \
        -errts ${subjdir}/Analysis/residual_error_series \
        -fout -tout \
        -xsave -x1D ${subjdir}/Analysis/xmat_rall.x1D -xjpeg ${subjdir}/Analysis/xmat_rall.jpg \
        -jobs 8 -float -overwrite \
        -bucket ${subjdir}/Analysis/pilotGLM_outbucket -cbucket ${subjdir}/Analysis/pilotGLM_cbucket
fi

