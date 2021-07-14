#!/bin/bash
# Convert INDIVRITL raw dicoms to bids format
# to upload to openneuro
## DEPENDENCIES: heudiconv + pydeface

# Taku Ito (taku.ito1@gmail.com)
# 6/8/21

# ALL subjects
allsubjects="013 014 016 017 018 021 023 024 026 027 028 030 031 032 033 034 035 037 038 039 040 041 042 043 045 046 047 048 049 050 053 055 056 057 058 062 063 066 067 068 069 070 072 074 075 076 077 081 085 086 087 088 090 092 093 094 095 097 098 099 101 102 103 104 105 106 108 109 110 111 112 114 115 117 119 120 121 122 123 124 125 126 127 128 129 130 131 132 134 135 136 137 138 139 140 141"

# Lower-case dicom files
subjects="013 014 016 017 018 021 023 024 026 027 031 035"
subjects="098 099 101 102 103 104 105 106 108 109 110 111 112 114 115 117 119 120 121 122 123 124 125 126 127 128 129 130 131 132 134 135 136 137 138 139 140 141"
##
execute=0
if [ $execute -eq 1 ]; then

    for subj in $subjects
    do
        echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        echo "==========CONVERTING TO BIDS + SUBJECT ${subj}============="
        echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"

        rm -rf /projects/IndivRITL/bids/.heudiconv/${subj}
        # t1 
        heudiconv -d /projects/IndivRITL/data/rawdata/{subject}/*t1_mpr_ns_sag_p2_iso_32Channel*/* \
            -o /projects/IndivRITL/bids \
            -f /projects/IndivRITL/bids/code/heuristic.py \
            -s $subj \
            --ses 01 \
            -c  dcm2niix -b

        rm -rf /projects/IndivRITL/bids/.heudiconv/${subj}
        # t2 
        heudiconv -d /projects/IndivRITL/data/rawdata/{subject}/*t2_spc_sag_p2_iso*/* \
            -o /projects/IndivRITL/bids \
            -f /projects/IndivRITL/bids/code/heuristic.py \
            -s $subj \
            --ses 01 \
            -c  dcm2niix -b

        rm -rf /projects/IndivRITL/bids/.heudiconv/${subj}
        # task BOLD
        heudiconv -d /projects/IndivRITL/data/rawdata/{subject}/*cmrr_mbep2d_bold_2mm_MB8_Task_8x*/* \
            -o /projects/IndivRITL/bids \
            -f /projects/IndivRITL/bids/code/heuristic.py \
            -s $subj \
            --ses 01 \
            -c  dcm2niix -b

        rm -rf /projects/IndivRITL/bids/.heudiconv/${subj}
        # rest BOLD
        heudiconv -d /projects/IndivRITL/data/rawdata/{subject}/*cmrr_mbep2d_bold_2mm_MB8_Rest*/* \
            -o /projects/IndivRITL/bids \
            -f /projects/IndivRITL/bids/code/heuristic.py \
            -s $subj \
            --ses 01 \
            -c  dcm2niix -b

        rm -rf /projects/IndivRITL/bids/.heudiconv/${subj}
        # spin echo maps 
        heudiconv -d /projects/IndivRITL/data/rawdata/{subject}/*cmrr_mbep2d_se_*/* \
            -o /projects/IndivRITL/bids \
            -f /projects/IndivRITL/bids/code/heuristic.py \
            -s $subj \
            --ses 01 \
            -c  dcm2niix -b


        rm -rf /projects/IndivRITL/bids/.heudiconv/${subj}
    done
fi

subjects="028 030 033 034 037 038 039 040 041 043 044 045 046 047 048 049 050 055 056 057 058 062 063 066 068 069 070 072 074 075 076 077 078 081 085 086 087 088 092 093 095 097" 
# capitalized  dicom files (IMA files)
execute=0
if [ $execute -eq 1 ]; then

    for subj in $subjects
    do
        echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        echo "==========CONVERTING TO BIDS + SUBJECT ${subj}============="
        echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"

        rm -rf /projects/IndivRITL/bids/.heudiconv/${subj}
        # t1 
        heudiconv -d /projects/IndivRITL/data/rawdata/{subject}/T1_MPR_NS_SAG_P2_ISO_32CHANNEL_*/*.IMA \
            -o /projects/IndivRITL/bids \
            -f /projects/IndivRITL/bids/code/heuristic.py \
            -s $subj \
            --ses 01 \
            -c  dcm2niix -b

        rm -rf /projects/IndivRITL/bids/.heudiconv/${subj}
        # t2 
        heudiconv -d /projects/IndivRITL/data/rawdata/{subject}/T2_SPC_SAG_P2_ISO_*/*.IMA \
            -o /projects/IndivRITL/bids \
            -f /projects/IndivRITL/bids/code/heuristic.py \
            -s $subj \
            --ses 01 \
            -c  dcm2niix -b

        rm -rf /projects/IndivRITL/bids/.heudiconv/${subj}
        # task BOLD
        heudiconv -d /projects/IndivRITL/data/rawdata/{subject}/CMRR_MBEP2D_BOLD_2MM_MB8_TASK_8X*/*.IMA \
            -o /projects/IndivRITL/bids \
            -f /projects/IndivRITL/bids/code/heuristic.py \
            -s $subj \
            --ses 01 \
            -c  dcm2niix -b

        rm -rf /projects/IndivRITL/bids/.heudiconv/${subj}
        # rest BOLD
        heudiconv -d /projects/IndivRITL/data/rawdata/{subject}/CMRR_MBEP2D_BOLD_2MM_MB8_REST*/*.IMA \
            -o /projects/IndivRITL/bids \
            -f /projects/IndivRITL/bids/code/heuristic.py \
            -s $subj \
            --ses 01 \
            -c  dcm2niix -b

        rm -rf /projects/IndivRITL/bids/.heudiconv/${subj}
        # spin echo maps 
        heudiconv -d /projects/IndivRITL/data/rawdata/{subject}/CMRR_MBEP2D_SE_*/*.IMA \
            -o /projects/IndivRITL/bids \
            -f /projects/IndivRITL/bids/code/heuristic.py \
            -s $subj \
            --ses 01 \
            -c  dcm2niix -b


        rm -rf /projects/IndivRITL/bids/.heudiconv/${subj}
    done
fi


### already ran on subject 013
subjects="014 016 017 018 021 023 024 026 027 028 030 031 032 033 034 035 037 038 039 040 041 042 043 045 046 047 048 049 050 053 055 056 057 058 062 063 066 067 068 069 070 072 074 075 076 077 081 085 086 087 088 090 092 093 094 095 097 098 099 101 102 103 104 105 106 108 109 110 111 112 114 115 117 119 120 121 122 123 124 125 126 127 128 129 130 131 132 134 135 136 137 138 139 140 141"
## Deface structural images in BIDs format
execute=1
if [ $execute -eq 1 ]; then
    for subj in $subjects
    do
        echo "**********DEFACING SUBJECT ${subj}************"
        anatdir=/projects/IndivRITL/bids/sub-${subj}/ses-01/anat/
        t1_nifti=$anatdir/sub-${subj}_ses-01_T1w.nii.gz
        t2_nifti=$anatdir/sub-${subj}_ses-01_T2w.nii.gz

        # Change permissions
        chmod 755 $t1_nifti
        chmod 755 $t2_nifti

        pydeface $t1_nifti --outfile $t1_nifti --force
        pydeface $t2_nifti --outfile $t2_nifti --force

    done
fi



## VERIFY EXISTENCE OF ALL SUBJECT DATA
execute=0
if [ $execute -eq 1 ]; then
    for subj in $allsubjects
    do
        echo "**********SUBJECT ${subj} VERIFICATION ************"
        subjdir=/projects/IndivRITL/bids/sub-${subj}/ses-01/
        anatdir=${subjdir}/anat/
        t1_nifti=$anatdir/sub-${subj}_ses-01_T1w.nii.gz
        t2_nifti=$anatdir/sub-${subj}_ses-01_T2w.nii.gz
        task1=${subjdir}/func/sub-${subj}_ses-01_task-cpro_run-01_bold.nii.gz
        task2=${subjdir}/func/sub-${subj}_ses-01_task-cpro_run-02_bold.nii.gz
        task3=${subjdir}/func/sub-${subj}_ses-01_task-cpro_run-03_bold.nii.gz
        task4=${subjdir}/func/sub-${subj}_ses-01_task-cpro_run-04_bold.nii.gz
        task5=${subjdir}/func/sub-${subj}_ses-01_task-cpro_run-05_bold.nii.gz
        task6=${subjdir}/func/sub-${subj}_ses-01_task-cpro_run-06_bold.nii.gz
        task7=${subjdir}/func/sub-${subj}_ses-01_task-cpro_run-07_bold.nii.gz
        task8=${subjdir}/func/sub-${subj}_ses-01_task-cpro_run-08_bold.nii.gz
        rest=${subjdir}/func/sub-${subj}_ses-01_task-rest_run-01_bold.nii.gz
        se_ap=${subjdir}/fmap/sub-${subj}_ses-01_acq-seAP_dir-AP_run-01_epi.nii.gz
        se_pa=${subjdir}/fmap/sub-${subj}_ses-01_acq-sePA_dir-PA_run-01_epi.nii.gz
        # Test for no task run 9
        task9=${subjdir}/func/sub-${subj}_ses-01_task-cpro_run-09_bold.nii.gz
        # 
        if [ ! -f "$t1_nifti" ]; then echo "$t1_nifti does not exists."; fi
        if [ ! -f "$t2_nifti" ]; then echo "$t2_nifti does not exists."; fi
        if [ ! -f "$task1" ]; then echo "$task1 does not exists."; fi
        if [ ! -f "$task2" ]; then echo "$task2 does not exists."; fi
        if [ ! -f "$task3" ]; then echo "$task3 does not exists."; fi
        if [ ! -f "$task4" ]; then echo "$task4 does not exists."; fi
        if [ ! -f "$task5" ]; then echo "$task5 does not exists."; fi
        if [ ! -f "$task6" ]; then echo "$task6 does not exists."; fi
        if [ ! -f "$task7" ]; then echo "$task7 does not exists."; fi
        if [ ! -f "$task8" ]; then echo "$task8 does not exists."; fi
        if [ -f "$task9" ]; then echo "ERROR!!!!! A 9TH TASK RUN DETECTED: $task9"; fi
        if [ ! -f "$rest" ]; then echo "$rest does not exists."; fi
        if [ ! -f "$se_ap" ]; then echo "$se_ap does not exists."; fi
        if [ ! -f "$se_pa" ]; then echo "$se_pa does not exists."; fi

    done
fi

## Remove sensitive files (with acquisition times) 
execute=0
if [ $execute -eq 1 ]; then
    for subj in $allsubjects
    do
        rm -f /projects/IndivRITL/bids/sub-${subj}/ses-01/sub-${subj}_ses-01_scans.json
        rm -f /projects/IndivRITL/bids/sub-${subj}/ses-01/sub-${subj}_ses-01_scans.tsv

    done
fi


