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

    --subj=<subjectID>          (required) The subject number 
    --dirSetUp=<"true">         (optional) Input "true", if you want to set up directory structure for this subject
    --runICAfix=<"true">        (optional) Run ICA Fix on resting-state data
    --runMSMAll=<"true">        (optional) Run MSMAll
EOF
    exit 1
}


opts_ShowVersionIfRequested $@

if opts_CheckForHelpRequest $@; then
    show_usage
fi


#Input Variables
subj=`opts_GetOpt1 "--subj" $@`
dirSetUp=`opts_GetOpt1 "--dirSetUp" $@`
runICAfix=`opts_GetOpt1 "--runICAfix" $@`
runMSMAll=`opts_GetOpt1 "--runMSMAll" $@`


#########################
# Set up HCP Environment - This is the customized environment for ColeLabLinux Server
HCPPipe=/usr/local/HCP_Pipelines_v2
EnvScript=${HCPPipe}/Examples/Scripts/SetUpHCPPipeline.sh
# Set up HCP Pipeline Environment
. ${EnvScript}
########################

# First need to reconstruct data

#########################
# Set up Subject Directory Parameters
basedir="/projects3/SRActFlow/"
datadir="${basedir}/data"
subjdir=${datadir}/${subj}
if [ ! -e $subjdir ]; then mkdir $subjdir; fi 


#########################


MSMAllTemplates=/usr/local/HCP_Pipelines_v2/global/templates/MSMAll/

#########################
# Pull out minimally preprocessed data
RestEPIs=(`ls ${rawdatadir}/ | grep mbep2d_bold_2mm_MB8_Rest_[0-9] || true;` ) # This returns an array of the scans delimited by spaces

#########################

if [ -z "$runICAfix" ]; then
    echo "Skipping ICA fix Node"
elif [ $runICAfix = true ]; then
    pushd $HCPPipe/ICAFIX/ 
fi

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
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

