#!/bin/bash

# Taku Ito
# 01/12/2017

# Resample 96k mesh surface grids to 64k

subjNums="101 102"
runNums="Rest1 Task1 Task2 Task3 Task4 Task5 Task6"

basedir=/projects3/StroopActFlow
datadir=${basedir}/data
ciftitemplate=/projects3/StroopActFlow/data/Q1-Q6_RelatedParcellation210.LR.CorticalAreas_dil_Colors.32k_fs_LR.dlabel.nii

for subj in $subjNums
do
    
    for run in $runNums
    do

        ciftiin=${datadir}/${subj}/MNINonLinear/Results/${run}/${run}_Atlas.dtseries.nii
        outdir=${datadir}/${subj}/analysis
        ciftiout=${outdir}/${run}_Atlas_64k.dtseries.nii
        
        echo "Resample ${run} for subject ${subj}"
        wb_command -cifti-resample $ciftiin COLUMN $ciftitemplate COLUMN ADAP_BARY_AREA CUBIC $ciftiout

    done
done
