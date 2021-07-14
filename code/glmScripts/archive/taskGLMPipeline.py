# Takuya Ito
# 03/28/2018

# Functions to run a GLM analysis

import numpy as np
import os
import glob
from nipy.modalities.fmri.hemodynamic_models import spm_hrf
import multiprocessing as mp
import statsmodels.api as sm
import h5py
import scipy.stats as stats
from scipy import signal
import constructDesignMatrices as dmat
import nibabel as nib

## Define GLOBAL variables (variables accessible to all functions
# Define base data directory
basedir = '/projects3/SRActFlow/data/'
datadir = basedir + 'postProcessing/'
# Define number of frames to skip
framesToSkip = 5
# Define list of subjects
subjNums = ['013','014','016','017','018','021','023','024','026','027','028','030','031','032','033','034','035','037','038','039','040','041','042','043','045','046','047','048','049','050','053','055','056','057','058','062','063','066','067','068','069','070','072','074','075','076','077','081','085','086','087','088','090','092','093','094','095','097','098','099','101','102','103','104','105','106','108','109','110','111','112','114','115','117','119','120','121','122','123','124','125','126','127','128','129','130','131','132','134','135','136','137','138','139','140','141']
# Define all runs you want to preprocess
allRuns = ['Task1', 'Task2', 'Task3', 'Task4', 'Task5', 'Task6', 'Task7', 'Task8']
taskRuns = ['Task1', 'Task2', 'Task3', 'Task4', 'Task5', 'Task6', 'Task7', 'Task8']

nRunsPerTask = 8
taskNames = ['Task1', 'Task2', 'Task3', 'Task4', 'Task5', 'Task6', 'Task7', 'Task8']
taskLength = 581
# Define the *output* directory for nuisance regressors
nuis_reg_dir = datadir + 'nuisanceRegressors/'
# Create directory if it doesn't exist
if not os.path.exists(nuis_reg_dir): os.makedirs(nuis_reg_dir)
# Define the directory containing the timing files
stimdir = '/projects3/SRActFlow/data/stimfiles/'
# Define the *output* directory for preprocessed data
outputdir = datadir + 'hcpPostProcCiric/' 
# TRlength
trLength = .785

def runGroupTaskGLM(task, nuisModel='24pXaCompCorXVolterra', taskModel='canonical', nproc=10):
    scount = 1
    for subj in subjNums:
        print 'Running task GLM for subject', subj, '|', scount, '/', len(subjNums), '... model:', taskModel
        taskGLM(subj, task, taskModel=taskModel, nuisModel=nuisModel, nproc=nproc)
        
        scount += 1

def runGroupRestGLM(nuisModel='24pXaCompCorXVolterra', taskModel='FIR', nproc=10):
    scount = 1
    for subj in subjNums:
        print 'Running task regression matrix on resting-state data for subject', subj, '|', scount, '/', len(subjNums), '... model:', taskModel
        taskGLM_onRest(subj, taskModel=taskModel, nuisModel=nuisModel, nproc=nproc)
        
        scount += 1

def taskGLM(subj, task, taskModel='canonical', nuisModel='24pXaCompCorXVolterra', nproc=8):
    """
    This function runs a task-based GLM on a single subject
    Will only regress out task-timing, using either a canonical HRF model or FIR model

    Input parameters:
        subj        :   subject number as a string
        task        :   fMRI task name (8 task runs)
        nuisModel   :   nuisance regression model (to identify input data)
        nproc       :   number of processes to use via multiprocessing
    """

    data_all = []
    for run in taskRuns:
        ## Load in data to be preprocessed - This needs to be a space x time 2d array
        inputfile = '/projects3/SRActFlow/data/' + subj + '/analysis/' + run + '_64kResampled.dtseries.nii'
        # Load data
        data = nib.load(inputfile).get_data()
        data = np.squeeze(data)

        tMask = np.ones((data.shape[0],))
        tMask[:framesToSkip] = 0

        # Skip frames
        data = data[framesToSkip:,:]
        # Demean each run
        data = signal.detrend(data,axis=0,type='constant')
        # Detrend each run
        data = signal.detrend(data,axis=0,type='linear')

        data_all.extend(data)

    data = np.asarray(data_all).T.copy()

    # Identify number of ROIs
    nROIs = data.shape[0]
    # Identify number of TRs
    nTRs = data.shape[1]

    # Load regressors for data
    X = loadTaskTiming(subj, task, taskModel='canonical', nRegsFIR=25)

    taskRegs = X['taskRegressors']
    # Load nuisance regressors
    nuisanceRegressors = loadNuisanceRegressors(subj,model='24pXaCompCorXVolterra',spikeReg=False)

    allRegs = np.hstack((nuisanceRegressors,taskRegs))

    inputs = []
    for roi in range(nROIs):
        inputs.append((data[roi,:], allRegs, True))

    print 'Running task GLM on subject', subj
    print '\tNumber of spatial dimensions:', nROIs
    print '\tNumber of task regressors:', taskRegs.shape[1]
    print '\tNumber of noise regressors:', nuisanceRegressors.shape[1]
    print '\tNumber of parallel processes:', nproc
    print '\tTask manipulation:', task
    pool = mp.Pool(processes=nproc)
    results = pool.map_async(_regression2,inputs).get()
    pool.close()
    pool.join()

    residual_ts = np.zeros((nROIs,nTRs))
    betas = np.zeros((nROIs,taskRegs.shape[1])) # All regressors, + constant regressors
    nTaskRegressors = int(taskRegs.shape[1])
    roi = 0
    for result in results:
        betas[roi,:] = result[0][-nTaskRegressors:]
        residual_ts[roi,:] = result[1]
        roi += 1
    
    h5f = h5py.File(outputdir + subj + '_glmOutput_data.h5','a')
    outname1 = 'taskRegression/' + task + '_' + nuisModel + '_taskReg_resid_' + taskModel
    outname2 = 'taskRegression/' + task + '_' + nuisModel + '_taskReg_betas_' + taskModel
    try:
        h5f.create_dataset(outname1,data=residual_ts)
        h5f.create_dataset(outname2,data=betas)
    except:
        del h5f[outname1], h5f[outname2]
        h5f.create_dataset(outname1,data=residual_ts)
        h5f.create_dataset(outname2,data=betas)
    h5f.close()

def loadTaskTiming(subj, task, taskModel='canonical', nRegsFIR=25):
    if task=='ALL':
        # rules
        logic = dmat.loadLogic(subj)
        sensory = dmat.loadSensory(subj)
        motor = dmat.loadMotor(subj)
        # stims
        colorStim = dmat.loadColorStim(subj)
        oriStim = dmat.loadOriStim(subj)
        pitchStim = dmat.loadPitchStim(subj)
        constantStim = dmat.loadConstantStim(subj)
        # sr interaction
        srRed = dmat.srRED(subj)
        srVert = dmat.srVERT(subj)
        srHigh = dmat.srHIGH(subj)
        srConstant = dmat.srCONSTANT(subj)
        # motor response
        motorResp = dmat.loadMotorResponse(subj)

        stimMat = np.hstack((logic,sensory,motor,colorStim,oriStim,pitchStim,constantStim,srRed,srVert,srHigh,srConstant,motorResp))
        stimIndex = []
        stimIndex.extend(np.repeat('Logic',4))
        stimIndex.extend(np.repeat('Sensory',4))
        stimIndex.extend(np.repeat('Motor',4))
        stimIndex.extend(np.repeat('colorStim',4))
        stimIndex.extend(np.repeat('oriStim',4))
        stimIndex.extend(np.repeat('pitchStim',4))
        stimIndex.extend(np.repeat('constantStim',4))
        stimIndex.extend(np.repeat('srRed',4))
        stimIndex.extend(np.repeat('srVertical',4))
        stimIndex.extend(np.repeat('srHigh',4))
        stimIndex.extend(np.repeat('srConstant',4))
        stimIndex.extend(np.repeat('motorResponse',4))

    if task=='betaSeries':
        filedir = stimdir + 'cproBetaSeries/'
        nBlocks = 128
        stimMat = []
        stimIndex = []
        # First add encoding blocks
        for block in range(1, nBlocks+1):
            enc_file = filedir + subj + '_stimfile_BetaSeries_EV1_Miniblock' + str(block) + '_Encoding.1D'
            stimMat.append(np.loadtxt(enc_file))
            stimIndex.append('Miniblock' + str(block) + '_Encoding')

        # Second add probe blocks
        for block in range(1, nBlocks+1):
            for probe in range(1,4): # 3 probes
                probe_file = filedir + subj + '_stimfile_BetaSeries_EV' + str(probe+1) + '_Miniblock' + str(block) + '_Probe' + str(probe) + '.1D'
                stimMat.append(np.loadtxt(probe_file))
                stimIndex.append('Miniblock' + str(block) + '_Probe' + str(probe))

        stimMat = np.asarray(stimMat).T
 

    nTRsPerRun = int(stimMat.shape[0]/nRunsPerTask)

    ## 
    if taskModel=='FIR':
        # Convolve taskstim regressors based on SPM canonical HRF (likely period of task-induced activity)

        ## First set up FIR design matrix
        taskStims_FIR = [] 
        for stim in range(stimMat.shape[1]):
            taskStims_FIR.append([])
            time_ind = np.where(stimMat[:,stim]==1)[0]
            blocks = _group_consecutives(time_ind) # Get blocks (i.e., sets of consecutive TRs)
            # Identify the longest block - set FIR duration to longest block
            maxRegsForBlocks = 0
            for block in blocks:
                if len(block) > maxRegsForBlocks: maxRegsForBlocks = len(block)
            taskStims_FIR[stim] = np.zeros((stimMat.shape[0],maxRegsForBlocks+nRegsFIR)) # Task timing for this condition is TR x length of block + FIR lag

        ## Now fill in FIR design matrix
        # Make sure to cut-off FIR models for each run separately
        trcount = 0
        for run in range(nRunsPerTask):
            trstart = trcount
            trend = trstart + nTRsPerRun
                
            for stim in range(stimMat.shape[1]):
                time_ind = np.where(stimMat[:,stim]==1)[0]
                blocks = _group_consecutives(time_ind) # Get blocks (i.e., sets of consecutive TRs)
                for block in blocks:
                    reg = 0
                    for tr in block:
                        # Set impulses for this run/task only
                        if trstart < tr < trend:
                            taskStims_FIR[stim][tr,reg] = 1
                            reg += 1

                        if not trstart < tr < trend: continue # If TR not in this run, skip this block

                    # If TR is not in this run, skip this block
                    if not trstart < tr < trend: continue

                    # Set lag due to HRF
                    for lag in range(1,nRegsFIR+1):
                        # Set impulses for this run/task only
                        if trstart < tr+lag < trend:
                            taskStims_FIR[stim][tr+lag,reg] = 1
                            reg += 1
            trcount += nTRsPerRun
        

        taskStims_FIR2 = np.zeros((stimMat.shape[0],1))
        task_index = []
        for stim in range(stimMat.shape[1]):
            task_index.extend(np.repeat(stim,taskStims_FIR[stim].shape[1]))
            taskStims_FIR2 = np.hstack((taskStims_FIR2,taskStims_FIR[stim]))

        taskStims_FIR2 = np.delete(taskStims_FIR2,0,axis=1)

        #taskRegressors = np.asarray(taskStims_FIR)
        taskRegressors = taskStims_FIR2
    
        # To prevent SVD does not converge error, make sure there are no columns with 0s
        zero_cols = np.where(np.sum(taskRegressors,axis=0)==0)[0]
        taskRegressors = np.delete(taskRegressors, zero_cols, axis=1)

    elif taskModel=='canonical':
        ## 
        # Convolve taskstim regressors based on SPM canonical HRF (likely period of task-induced activity)
        taskStims_HRF = np.zeros(stimMat.shape)
        spm_hrfTS = spm_hrf(trLength,oversampling=1)
       
        trcount = 0
        for run in range(nRunsPerTask):
            trstart = trcount
            trend = trstart + nTRsPerRun

            for stim in range(stimMat.shape[1]):

                # Perform convolution
                tmpconvolve = np.convolve(stimMat[trstart:trend,stim],spm_hrfTS)
                tmpconvolve_run = tmpconvolve[:nTRsPerRun] # Make sure to cut off at the end of the run
                taskStims_HRF[trstart:trend,stim] = tmpconvolve_run

            trcount += nTRsPerRun

        taskRegressors = taskStims_HRF.copy()
    
    # Create temporal mask (skipping which frames?)
    tMask = []
    for run in range(nRunsPerTask):
        tmp = np.ones((nTRsPerRun,), dtype=bool)
        tmp[:framesToSkip] = False
        tMask.extend(tmp)
    tMask = np.asarray(tMask,dtype=bool)

    output = {}
    # Commented out since we demean each run prior to loading data anyway
    output['taskRegressors'] = taskRegressors[tMask,:]
    output['taskDesignMat'] = stimMat[tMask,:]
    output['stimIndex'] = stimIndex

    return output

def loadNuisanceRegressors(subj,model='24pXaCompCorXVolterra',spikeReg=False):
    """
    LOAD and concatenate all nuisance regressors across all tasks
    """

    concatNuisRegressors = []
    # Load nuisance regressors for this data
    h5f = h5py.File(nuis_reg_dir + subj + '_nuisanceRegressors.h5','r') 
    for run in taskRuns:
        
        if model=='24pXaCompCorXVolterra':
            # Motion parameters + derivatives
            motion_parameters = h5f[run]['motionParams'][:].copy()
            motion_parameters_deriv = h5f[run]['motionParams_deriv'][:].copy()
            # WM aCompCor + derivatives
            aCompCor_WM = h5f[run]['aCompCor_WM'][:].copy()
            aCompCor_WM_deriv = h5f[run]['aCompCor_WM_deriv'][:].copy()
            # Ventricles aCompCor + derivatives
            aCompCor_ventricles = h5f[run]['aCompCor_ventricles'][:].copy()
            aCompCor_ventricles_deriv = h5f[run]['aCompCor_ventricles_deriv'][:].copy()
            # Create nuisance regressors design matrix
            nuisanceRegressors = np.hstack((motion_parameters, motion_parameters_deriv, aCompCor_WM, aCompCor_WM_deriv, aCompCor_ventricles, aCompCor_ventricles_deriv))
            quadraticRegressors = nuisanceRegressors**2
            nuisanceRegressors = np.hstack((nuisanceRegressors,quadraticRegressors))
        
        elif model=='18p':
            # Motion parameters + derivatives
            motion_parameters = h5f[run]['motionParams'][:].copy()
            motion_parameters_deriv = h5f[run]['motionParams_deriv'][:].copy()
            # Global signal + derivatives
            global_signal = h5f[run]['global_signal'][:].copy()
            global_signal_deriv = h5f[run]['global_signal_deriv'][:].copy()
            # white matter signal + derivatives
            wm_signal = h5f[run]['wm_signal'][:].copy()
            wm_signal_deriv = h5f[run]['wm_signal_deriv'][:].copy()
            # ventricle signal + derivatives
            ventricle_signal = h5f[run]['ventricle_signal'][:].copy()
            ventricle_signal_deriv = h5f[run]['ventricle_signal_deriv'][:].copy()
            # Create nuisance regressors design matrix
            tmp = np.vstack((global_signal,global_signal_deriv,wm_signal,wm_signal_deriv,ventricle_signal,ventricle_signal_deriv)).T # Need to vstack, since these are 1d arrays
            nuisanceRegressors = np.hstack((motion_parameters, motion_parameters_deriv, tmp))

        elif model=='16pNoGSR':
            # Motion parameters + derivatives
            motion_parameters = h5f[run]['motionParams'][:].copy()
            motion_parameters_deriv = h5f[run]['motionParams_deriv'][:].copy()
            # white matter signal + derivatives
            wm_signal = h5f[run]['wm_signal'][:].copy()
            wm_signal_deriv = h5f[run]['wm_signal_deriv'][:].copy()
            # ventricle signal + derivatives
            ventricle_signal = h5f[run]['ventricle_signal'][:].copy()
            ventricle_signal_deriv = h5f[run]['ventricle_signal_deriv'][:].copy()
            # Create nuisance regressors design matrix
            tmp = np.vstack((wm_signal,wm_signal_deriv,ventricle_signal,ventricle_signal_deriv)).T # Need to vstack, since these are 1d arrays
            nuisanceRegressors = np.hstack((motion_parameters, motion_parameters_deriv, tmp))
        
        elif model=='12pXaCompCor':
            # Motion parameters + derivatives
            motion_parameters = h5f[run]['motionParams'][:].copy()
            motion_parameters_deriv = h5f[run]['motionParams_deriv'][:].copy()
            # WM aCompCor + derivatives
            aCompCor_WM = h5f[run]['aCompCor_WM'][:].copy()
            aCompCor_WM_deriv = h5f[run]['aCompCor_WM_deriv'][:].copy()
            # Ventricles aCompCor + derivatives
            aCompCor_ventricles = h5f[run]['aCompCor_ventricles'][:].copy()
            aCompCor_ventricles_deriv = h5f[run]['aCompCor_ventricles_deriv'][:].copy()
            # Create nuisance regressors design matrix
            nuisanceRegressors = np.hstack((motion_parameters, motion_parameters_deriv, aCompCor_WM, aCompCor_WM_deriv, aCompCor_ventricles, aCompCor_ventricles_deriv))
        
        elif model=='36p':
            # Motion parameters + derivatives
            motion_parameters = h5f[run]['motionParams'][:].copy()
            motion_parameters_deriv = h5f[run]['motionParams_deriv'][:].copy()
            # Global signal + derivatives
            global_signal = h5f[run]['global_signal'][:].copy()
            global_signal_deriv = h5f[run]['global_signal_deriv'][:].copy()
            # white matter signal + derivatives
            wm_signal = h5f[run]['wm_signal'][:].copy()
            wm_signal_deriv = h5f[run]['wm_signal_deriv'][:].copy()
            # ventricle signal + derivatives
            ventricle_signal = h5f[run]['ventricle_signal'][:].copy()
            ventricle_signal_deriv = h5f[run]['ventricle_signal_deriv'][:].copy()
            # Create nuisance regressors design matrix
            tmp = np.vstack((global_signal,global_signal_deriv,wm_signal,wm_signal_deriv,ventricle_signal,ventricle_signal_deriv)).T # Need to vstack, since these are 1d arrays
            nuisanceRegressors = np.hstack((motion_parameters, motion_parameters_deriv, tmp))
            quadraticRegressors = nuisanceRegressors**2
            nuisanceRegressors = np.hstack((nuisanceRegressors,quadraticRegressors))


        if spikeReg:
            # Obtain motion spikes
            try:
                motion_spikes = h5f[run]['motionSpikes'][:].copy()
                nuisanceRegressors = np.hstack((nuisanceRegressors,motion_spikes))
            except:
                print 'Spike regression option was chosen... but no motion spikes for subj', subj, '| run', run, '!'
            # Update the model name - to keep track of different model types for output naming
            model = model + '_spikeReg' 

        concatNuisRegressors.extend(nuisanceRegressors[framesToSkip:,:].copy())

    h5f.close()
    nuisanceRegressors = np.asarray(concatNuisRegressors)

    return nuisanceRegressors

def _regression2((data,regressors,constant)):
    """
    Hand coded OLS regression using closed form equation: betas = (X'X)^(-1) X'y
    """
    # Add 'constant' regressor
    if constant:
        regressors = sm.add_constant(regressors)
    X = regressors.copy()
    try:
#        #C_ss_inv = np.linalg.inv(np.dot(X.T,X))
        C_ss_inv = np.linalg.pinv(np.dot(X.T,X))
    except np.linalg.LinAlgError as err:
        C_ss_inv = np.linalg.pinv(np.cov(X.T))
    betas = np.dot(C_ss_inv,np.dot(X.T,data.T))
    resid = data - (betas[0] + np.dot(X[:,1:],betas[1:])).T
    return betas, resid

def _group_consecutives(vals, step=1):
    """Return list of consecutive lists of numbers from vals (number list)."""
    run = []
    result = [run]
    expect = None
    for v in vals:
        if (v == expect) or (expect is None):
            run.append(v)
        else:
            run = [v]
            result.append(run)
        expect = v + step
    return result

