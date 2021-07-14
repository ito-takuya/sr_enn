#!/bin/bash

# Taku Ito
# 02/22/2018

# Resample 92k surface coordinates to 64k surface coordinates for Glasser parcels


listOfSubjects="013 014 016 017 018 021 023 024 025 026 027 028 030 031 032 033 034 035 037 038 039 040 041 042 043 045 046 047 048 049 050 053 055 056 057 058 062 063 064 066 067 068 069 070 072 074 075 076 077 081 082 085 086 087 088 090 092 093 094 095 097 098 099 101 102 103 104 105 106 108 109 110 111 112 114 115 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 134 135 136 137 138 139 140 141"
listOfSubjects="038"
dataFiles="Rest1 Task1 Task2 Task3 Task4 Task5 Task6 Task7 Task8"
basedir=/projects3/SRActFlow/data/
inputdir=/projects/IndivRITL/data/

resampleAtlas=/projects/AnalysisTools/ParcelsGlasser2016/Q1-Q6_RelatedParcellation210.LR.CorticalAreas_dil_Colors.32k_fs_RL.dlabel.nii

for subj in $listOfSubjects
do
    echo "Resampling data for subject ${subj}"
    subjdir=${basedir}/${subj}/analysis
    if [ ! -e $subjdir ]; then mkdir -p $subjdir; fi
    
    for run in $dataFiles
    do
        echo "Subject ${subj}, Run ${run}"
        inputFile=${inputdir}/${subj}/MNINonLinear/Results/${run}/${run}_Atlas.dtseries.nii
        outputfile=${subjdir}/${run}_64kResampled.dtseries.nii
        wb_command -cifti-resample $inputFile COLUMN $resampleAtlas COLUMN ADAP_BARY_AREA CUBIC $outputfile
    done
done
