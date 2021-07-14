#!/bin/bash
# Adapted from Mike's TaskFCMethods project for ModalityControl
#Script to create a parcellated CIFTI file, based on the Glasser2016 parcellation

basedir=/projects3/StroopActFlow/
datadir=${basedir}/data/
subjList="101 102" # 101
runNameList="Rest1 Task1 Task2 Task3 Task4 Task5 Task6"
parcelFile=/projects3/StroopActFlow/data/Q1-Q6_RelatedParcellation210.LR.CorticalAreas_dil_Colors.32k_fs_LR.dlabel.nii

for subjNum in $subjList; do
  subjDir=${datadir}/${subjNum}/
  subjDirOutput=/projects3/StroopActFlow/data/${subjNum}
  if [ "`ls -d $subjDirOutput`" == "" ]; then mkdir $subjDirOutput; fi	#Making directory if it doesn't exist yet
  subjAnalysisDir=${subjDirOutput}/analysis/
  if [ "`ls -d $subjAnalysisDir`" == "" ]; then mkdir $subjAnalysisDir; fi	#Making directory if it doesn't exist yet

  for runName in $runNameList; do
    subjRunDir=${subjDir}/MNINonLinear/Results/${runName}/
    pushd ${subjRunDir}

    inputFileName=${runName}_Atlas.dtseries.nii

    #Run commands to parcellate data file
    wb_command -cifti-parcellate ${inputFileName} ${parcelFile} COLUMN ${subjAnalysisDir}/${runName}_Atlas.LR.Glasser2016Parcels.32k_fs_LR.ptseries.nii -method MEAN

    popd
  done

done
