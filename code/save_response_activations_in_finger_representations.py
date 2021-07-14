# Taku Ito
# 2/22/2019
# General function modules for SRActFlow
# For group-level/cross-subject analyses

import numpy as np
import multiprocessing as mp
import scipy.stats as stats
import nibabel as nib
import statsmodels.api as sm
import sklearn
import h5py
import os
os.sys.path.append('glmScripts/')
import taskGLMPipeline_v2 as tgp
import sys
sys.path.append('utils/')
import loadExperimentalData as led
import tools


projectdir = '/home/ti61/f_mc1689_1/SRActFlow/'

glasserfile2 = projectdir + 'data/Q1-Q6_RelatedParcellation210.LR.CorticalAreas_dil_Colors.32k_fs_RL.dlabel.nii'
glasser2 = nib.load(glasserfile2).get_data()
glasser2 = np.squeeze(glasser2)

subjNums = ['013','014','016','017','018','021','023','024','026','027','028','030','031','032','033',
            '034','035','037','038','039','040','041','042','043','045','046','047','048','049','050',
            '053','055','056','057','058','062','063','066','067','068','069','070','072','074','075',
            '076','077','081','085','086','087','088','090','092','093','094','095','097','098','099',
            '101','102','103','104','105','106','108','109','110','111','112','114','115','117','119',
            '120','121','122','123','124','125','126','127','128','129','130','131','132','134','135',
            '136','137','138','139','140','141']

###############################################
# Begin script

#### Load original data
print('Load original motor response data')
nResponses = 2
data_task_rh = np.zeros((len(glasser2),nResponses,len(subjNums)))
data_task_lh = np.zeros((len(glasser2),nResponses,len(subjNums)))

scount = 0
for subj in subjNums:
    data_task_rh[:,:,scount] = np.real(tools.loadMotorResponses(subj,hand='Right'))
    data_task_lh[:,:,scount] = np.real(tools.loadMotorResponses(subj,hand='Left'))
    scount += 1


####
# Isolate RH and LH vertices for motor response betas
tmp = np.squeeze(nib.load(projectdir + 'data/results/MAIN/MotorRegionsMasksPerSubj/sractflow_smn_outputRH_mask.dscalar.nii').get_data())
rh_ind = np.where(tmp==True)[0]
realdata_rh = data_task_rh[rh_ind,:,:].copy()

tmp = np.squeeze(nib.load(projectdir + 'data/results/MAIN/MotorRegionsMasksPerSubj/sractflow_smn_outputLH_mask.dscalar.nii').get_data())
lh_ind = np.where(tmp==True)[0]
realdata_lh = data_task_lh[lh_ind,:,:].copy()

h5f = h5py.File(projectdir + 'data/results/MAIN/MotorResponseBetas_OutputVertices.h5','a')
h5f.create_dataset('RH',data=realdata_rh)
h5f.create_dataset('LH',data=realdata_lh)
h5f.close()
