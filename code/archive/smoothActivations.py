# Taku Ito
# 03/26/2019
# Script to smooth vertex-wise activations 

import numpy as np
import utils.mapToSurface as mts
import tools_group
import os
import nibabel as nib
import utils.permutationTesting as pt
pt = reload(pt)

subjNums = ['013','014','016','017','018','021','023','024','026','027','028','030','031','032','033',
            '034','035','037','038','039','040','041','042','043','045','046','047','048','049','050',
            '053','055','056','057','058','062','063','066','067','068','069','070','072','074','075',
            '076','077','081','085','086','087','088','090','092','093','094','095','097','098','099',
            '101','102','103','104','105','106','108','109','110','111','112','114','115','117','119',
            '120','121','122','123','124','125','126','127','128','129','130','131','132','134','135',
            '136','137','138','139','140','141']

def smoothInputBetas():
    inputtypes = ['colorStim','oriStim','pitchStim','constantStim']
    for subj in subjNums:
        print 'Smoothing input betas for subject', subj
        for inputtype in inputtypes:
            filename = subj + '_' + inputtype + '_glmBetaActivations'
            filename_smoothed = subj + '_' + inputtype + '_glmBetaActivations_Smoothed'
            subjdir = '/projects3/SRActFlow/data/cpro2/' + subj + '/MNINonLinear/fsaverage_LR32k/'
            lsurf = subjdir + subj + '.L.inflated.32k_fs_LR.surf.gii'
            rsurf = subjdir + subj + '.R.inflated.32k_fs_LR.surf.gii'
            filedir = '/projects3/SRActFlow/data/results/glmActivationsSmoothing/'

            data = tools_group.loadInputActivity(subj,inputtype) # load input data
            data = data.astype(float)
            mts.mapToSurface(data,filename=filedir + filename) # map to surface
            wb_command = 'wb_command -cifti-smoothing ' + filedir + filename + '.dscalar.nii 4 4 COLUMN ' + filedir + filename + '_smoothed.dscalar.nii'
            wb_command += ' -left-surface ' + lsurf
            wb_command += ' -right-surface ' + rsurf
            os.system(wb_command) # run workbench command

def smoothMotorBetas():
    conds = ['Right', 'Left']
    for subj in subjNums:
        print 'Smoothing input betas for subject', subj
        for cond in conds:
            filename = subj + '_' + cond + '_glmBetaActivations'
            filename_smoothed = subj + '_' + cond + '_glmBetaActivations_Smoothed'
            subjdir = '/projects3/SRActFlow/data/cpro2/' + subj + '/MNINonLinear/fsaverage_LR32k/'
            lsurf = subjdir + subj + '.L.inflated.32k_fs_LR.surf.gii'
            rsurf = subjdir + subj + '.R.inflated.32k_fs_LR.surf.gii'
            filedir = '/projects3/SRActFlow/data/results/glmActivationsSmoothing/'

            data = tools_group.loadMotorResponses(subj,cond) # load input data
            data = data.astype(float)
            mts.mapToSurface(data,filename=filedir + filename) # map to surface
            wb_command = 'wb_command -cifti-smoothing ' + filedir + filename + '.dscalar.nii 4 4 COLUMN ' + filedir + filename + '_smoothed.dscalar.nii'
            wb_command += ' -left-surface ' + lsurf
            wb_command += ' -right-surface ' + rsurf
            os.system(wb_command) # run workbench command

def smoothRuleBetas():
    conds = ['Logic', 'Sensory', 'Motor']
    for subj in subjNums:
        print 'Smoothing input betas for subject', subj
        for cond in conds:
            filename = subj + '_' + cond + '_glmBetaActivations'
            filename_smoothed = subj + '_' + cond + '_glmBetaActivations_Smoothed'
            subjdir = '/projects3/SRActFlow/data/cpro2/' + subj + '/MNINonLinear/fsaverage_LR32k/'
            lsurf = subjdir + subj + '.L.inflated.32k_fs_LR.surf.gii'
            rsurf = subjdir + subj + '.R.inflated.32k_fs_LR.surf.gii'
            filedir = '/projects3/SRActFlow/data/results/glmActivationsSmoothing/'

            data = tools_group.loadRuleEncoding(subj,cond) # load input data
            data = data.astype(float)
            mts.mapToSurface(data,filename=filedir + filename) # map to surface
            wb_command = 'wb_command -cifti-smoothing ' + filedir + filename + '.dscalar.nii 4 4 COLUMN ' + filedir + filename + '_smoothed.dscalar.nii'
            wb_command += ' -left-surface ' + lsurf
            wb_command += ' -right-surface ' + rsurf
            os.system(wb_command) # run workbench command

def smoothSRBetas():
    conds = ['srRed', 'srVertical', 'srHigh', 'srConstant']
    for subj in subjNums:
        print 'Smoothing input betas for subject', subj
        for cond in conds:
            filename = subj + '_' + cond + '_glmBetaActivations'
            filename_smoothed = subj + '_' + cond + '_glmBetaActivations_Smoothed'
            subjdir = '/projects3/SRActFlow/data/cpro2/' + subj + '/MNINonLinear/fsaverage_LR32k/'
            lsurf = subjdir + subj + '.L.inflated.32k_fs_LR.surf.gii'
            rsurf = subjdir + subj + '.R.inflated.32k_fs_LR.surf.gii'
            filedir = '/projects3/SRActFlow/data/results/glmActivationsSmoothing/'

            data = tools_group.loadInputActivity(subj,cond) # load input data
            data = data.astype(float)
            mts.mapToSurface(data,filename=filedir + filename) # map to surface
            wb_command = 'wb_command -cifti-smoothing ' + filedir + filename + '.dscalar.nii 4 4 COLUMN ' + filedir + filename + '_smoothed.dscalar.nii'
            wb_command += ' -left-surface ' + lsurf
            wb_command += ' -right-surface ' + rsurf
            os.system(wb_command) # run workbench command


def runTTestOnData(smoothed=True):
    conds = ['Right','Left']
    for cond in conds:
        print 'Running maxT test on condition', cond
        print '\tLoading subject data...'
        data_matrix = []
        for subj in subjNums:
            filedir = '/projects3/SRActFlow/data/results/glmActivationsSmoothing/'
            if smoothed:
                filename = subj + '_' + cond + '_glmBetaActivations_smoothed'
                outname = 'maxT_' + cond + '_glmBetaActivations_smoothed'
            else:
                filename = subj + '_' + cond + '_glmBetaActivations'
                outname = 'maxT_' + cond + '_glmBetaActivations'

            # Load data
            data = np.squeeze(nib.load(filedir + filename + '.dscalar.nii').get_data()).T
            diff_arr = data[:,1] - data[:,0]
            data_matrix.append(diff_arr)

        print '\tRunning maxT permutations...'
        data_matrix = np.asarray(data_matrix).T
        t_arr, top_maxt, bot_maxt = pt.maxT(data_matrix, nullmean=0, alpha=.05, tail=0, permutations=1000, nproc=20, pvals=False)
        print bot_maxt
        print top_maxt
        print 'max t emp', np.max(t_arr)
        print 'min t emp', np.min(t_arr)
        t_arr_thresh = np.multiply(t_arr, t_arr>top_maxt) + np.multiply(t_arr, t_arr<bot_maxt)
        out_arr = np.zeros((len(t_arr_thresh),2))
        out_arr[:,0] = t_arr_thresh
        out_arr[:,1] = t_arr
        print '\tNum significant vertices:', np.sum(t_arr_thresh!=0)
        mts.mapToSurface(out_arr,filedir+outname)
            
