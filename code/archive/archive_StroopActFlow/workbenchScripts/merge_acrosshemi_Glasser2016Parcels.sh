#!/bin/bash

basedir=/projects/AnalysisTools/ParcelsGlasser2016/

leftHemi=/projects/AnalysisTools/ParcelsGlasser2016/Q1-Q6_RelatedParcellation210/MNINonLinear/fsaverage_LR32k/Q1-Q6_RelatedParcellation210.L.CorticalAreas_dil_Colors.32k_fs_LR.dlabel.nii
rightHemi=/projects/AnalysisTools/ParcelsGlasser2016/Q1-Q6_RelatedParcellation210/MNINonLinear/fsaverage_LR32k/Q1-Q6_RelatedParcellation210.R.CorticalAreas_dil_Colors.32k_fs_LR.dlabel.nii

#Merge label files
#wb_command -cifti-merge Q1-Q6_RelatedParcellation210.LR.CorticalAreas_dil_Colors.32k_fs_LR.label.nii -cifti $leftHemi -cifti $rightHemi

#wb_command -cifti-merge-parcels COLUMN Q1-Q6_RelatedParcellation210.LR.CorticalAreas_dil_Colors.32k_fs_LR.label.nii -cifti $leftHemi -cifti $rightHemi

wb_command -cifti-merge-dense COLUMN /projects3/StroopActFlow/data/Q1-Q6_RelatedParcellation210.LR.CorticalAreas_dil_Colors.32k_fs_LR.dlabel.nii -cifti $leftHemi -cifti $rightHemi

wb_command -cifti-merge-dense COLUMN /projects3/StroopActFlow/data/Q1-Q6_RelatedParcellation210.RL.CorticalAreas_dil_Colors.32k_fs_RL.dlabel.nii -cifti $rightHemi -cifti $leftHemi

