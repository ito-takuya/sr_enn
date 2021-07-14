# Taku Ito
# 03/01/2019
# Compute PCA FC 

import numpy as np
import nibabel as nib
import os
import h5py
os.environ['OMP_NUM_THREADS'] = str(1)
import multiprocessing as mp
import scipy.stats as stats
from scipy import signal
import time
from sklearn.decomposition import PCA
import tools_group_rsa as tools_group


#### Set up base parameters
# Excluding 084
subjNums = ['013','014','016','017','018','021','023','024','026','027','028','030','031','032','033',
            '034','035','037','038','039','040','041','042','043','045','046','047','048','049','050',
            '053','055','056','057','058','062','063','066','067','068','069','070','072','074','075',
            '076','077','081','085','086','087','088','090','092','093','094','095','097','098','099',
            '101','102','103','104','105','106','108','109','110','111','112','114','115','117','119',
            '120','121','122','123','124','125','126','127','128','129','130','131','132','134','135',
            '136','137','138','139','140','141']



basedir = '/projects3/SRActFlow/'

## General parameters/variables
nParcels = 360
nSubjs = len(subjNums)

glasserfile2 = '/projects/AnalysisTools/ParcelsGlasser2016/Q1-Q6_RelatedParcellation210.LR.CorticalAreas_dil_Colors.32k_fs_RL.dlabel.nii'
glasser2 = nib.load(glasserfile2).get_data()
glasser2 = np.squeeze(glasser2)
####



def computeInputStimToHiddenFC(inputtype,n_hiddenregions=10,n_components=500,nproc=20):
    scount = 1
    for subj in subjNums:
        print 'Running on subject', subj, '(', scount, '/', len(subjNums), ')'
        outputdir = '/projects3/SRActFlow/data/results/GroupfMRI/RSA/LayerToLayerFC/'
        outputfilename = outputdir + '/' + inputtype + 'To' + 'HiddenLayer_TaskFC_subj' + subj + '.h5'
        sourcedir = '/projects3/SRActFlow/data/results/GroupfMRI/InputStimuliDecoding/'
        input_regions = np.loadtxt(sourcedir + 'InputStimuliRegions_' + inputtype + '.csv',delimiter=',')
        targetdir = '/projects3/SRActFlow/data/results/GroupfMRI/RSA/'
        hiddenregions = np.loadtxt(targetdir + 'RSA_Similarity_SortedRegions.txt',delimiter=',')
        hiddenregions = hiddenregions[:n_hiddenregions]

        data = tools_group.loadTaskResiduals(subj,model='24pXaCompCorXVolterra',zscore=False)
        # Demean
        data = signal.detrend(data,axis=1,type='constant')

        layerToLayerFC(data,input_regions,hiddenregions,outputfilename,n_components=n_components,nproc=nproc)

        scount += 1

def computeRuleToHiddenFC(ruletype,n_hiddenregions=10,n_components=500,nproc=20):
    scount = 1
    for subj in subjNums:
        print 'Running on subject', subj, '(', scount, '/', len(subjNums), '):', ruletype
        outputdir = '/projects3/SRActFlow/data/results/GroupfMRI/RSA/LayerToLayerFC/'
        outputfilename = outputdir + '/' + ruletype + 'RuleTo' + 'HiddenLayer_TaskFC_subj' + subj + '.h5'
        sourcedir = '/projects3/SRActFlow/data/results/GroupfMRI/RuleDecoding/'
        rule_regions = np.loadtxt(sourcedir + ruletype + 'Rule_Regions.csv',delimiter=',')
        targetdir = '/projects3/SRActFlow/data/results/GroupfMRI/RSA/'
        hiddenregions = np.loadtxt(targetdir + 'RSA_Similarity_SortedRegions.txt',delimiter=',')
        hiddenregions = hiddenregions[:n_hiddenregions]

        data = tools_group.loadTaskResiduals(subj,model='24pXaCompCorXVolterra',zscore=False)
        # Demean
        data = signal.detrend(data,axis=1,type='constant')

        layerToLayerFC(data,rule_regions,hiddenregions,outputfilename,n_components=n_components,nproc=nproc)

        scount += 1

def computeHiddenLayerToMotorResponseFC(n_hiddenregions=10,n_components=500,nproc=20):
    scount = 1
    for subj in subjNums:
        print 'Running on subject', subj, '(', scount, '/', len(subjNums), ')'
        outputdir = '/projects3/SRActFlow/data/results/GroupfMRI/RSA/LayerToLayerFC/'
        outputfilename = outputdir + '/HiddenLayerToMotorResponse_TaskFC_subj' + subj + '.h5'
        sourcedir = '/projects3/SRActFlow/data/results/GroupfMRI/RSA/'
        hiddenregions = np.loadtxt(sourcedir + 'RSA_Similarity_SortedRegions.txt',delimiter=',')
        hiddenregions = hiddenregions[:n_hiddenregions]
        targetdir = '/projects3/SRActFlow/data/results/GroupfMRI/MotorResponseDecoding/'
        motor_resp_regions_LH = np.loadtxt(targetdir + 'MotorResponseRegions_LH.csv',delimiter=',')
        motor_resp_regions_RH = np.loadtxt(targetdir + 'MotorResponseRegions_RH.csv',delimiter=',')

        motor_resp_regions = np.hstack((motor_resp_regions_LH,motor_resp_regions_RH))
        #motor_resp_regions = [9,189]


        data = tools_group.loadTaskResiduals(subj,model='24pXaCompCorXVolterra',zscore=False)
        # Demean
        data = signal.detrend(data,axis=1,type='constant')

        layerToLayerFC(data,hiddenregions,motor_resp_regions,outputfilename,n_components=n_components,nproc=nproc)

        scount += 1

def pcaFC(stim,resp,n_components=500,constant=False,nproc=10):
    """
    stim    - time x feature/region matrix of regressors
    resp    - time x feature/region matrix of targets (y-values)
    """
    print '\tRunning PCA'
    os.environ['OMP_NUM_THREADS'] = str(nproc)
    if stim.shape[1]<n_components:
        n_components = stim.shape[1]
    pca = PCA(n_components)
    reduced_mat = pca.fit_transform(stim) # Time X Features
    components = pca.components_
    
    print '\tRunning regression'
    betas, resid = regression(resp,reduced_mat,alpha=0,constant=constant) # betas are components x targets 
    
    ## Remove coliders
    # Identify pair-wise covariance matrix
    cov_mat = np.dot(reduced_mat.T, resp)
    # Identify postive weights with also postive cov
    pos_mat = np.multiply(cov_mat>0,betas>0)
    # Identify negative weights with also negative cov
    neg_mat = np.multiply(cov_mat<0,betas<0)
    # Now identify both positive and negative weights
    pos_weights = np.multiply(pos_mat,betas)
    neg_weights = np.multiply(neg_mat,betas)
    fc_mat = pos_weights + neg_weights
    
    # Now map back into physical vertex space
    # Dimensions: Source X Target vertices
    #fc_mat = np.dot(fc_mat.T,components).T
 
    return fc_mat,components 

def layerToLayerFC(data,sourceROIs,targetROIs,filename,n_components=500,nproc=10):
    """
    First identify if the source and target ROIs have any overlapping ROIs, and remove them from each set
    Then, 
        For some set of source ROIs, identify the vertices and concatenate them
        For some set of target ROIs, identify the vertices and concatenate them
    Then run PCA regression to find the weights
    
    PARAMETERS:
        data        :   resting-state data
        sourceROIs  :   source ROIs
        targetROIs  :   target ROIs
        filename    :   string for the filename to save data
        n_components:   Number of components for PC regression
        nproc       :   Number of processes to use in parallel

    """
    ####
    # Step 1 - remove overlapping ROIs
    overlappingROIs = np.intersect1d(sourceROIs,targetROIs)
    unique_sources = []
    for roi in sourceROIs:
        if roi in overlappingROIs:
            continue
        else:
            unique_sources.append(roi)
    
    unique_targets = []
    for roi in targetROIs:
        #if roi in overlappingROIs:
        #    continue
        #else:
        unique_targets.append(roi)

    ####
    # Step 2 - concatenate data for unique sources/targets
    sourcemat = []
    for roi in unique_sources:
        roi_ind = np.where(glasser2==roi+1)[0]
        sourcemat.extend(data[roi_ind,:])
    sourcemat = np.asarray(sourcemat).T
    
    targetmat = []
    for roi in unique_targets:
        roi_ind = np.where(glasser2==roi+1)[0]
        targetmat.extend(data[roi_ind,:])
    targetmat = np.asarray(targetmat).T


    ####
    # Step 3 - run PCA regression
    sourceToTargetMappings, eigenvectors = pcaFC(sourcemat,targetmat,n_components=n_components,nproc=nproc)
    # Save out to file
    print '\tSaving out to disk'
    h5f = h5py.File(filename,'a')
    try:
        h5f.create_dataset('sourceToTargetMapping',data=sourceToTargetMappings)
        h5f.create_dataset('eigenvectors',data=eigenvectors)
    except:
        del h5f['sourceToTargetMapping'], h5f['eigenvectors']
        h5f.create_dataset('sourceToTargetMapping',data=sourceToTargetMappings)
        h5f.create_dataset('eigenvectors',data=eigenvectors)
    h5f.close()

def regression(data,regressors,alpha=0,constant=True):
    """
    Taku Ito
    2/21/2019

    Hand coded OLS regression using closed form equation: betas = (X'X + alpha*I)^(-1) X'y
    Set alpha = 0 for regular OLS.
    Set alpha > 0 for ridge penalty

    PARAMETERS:
        data = observation x feature matrix (e.g., time x regions)
        regressors = observation x feature matrix
        alpha = regularization term. 0 for regular multiple regression. >0 for ridge penalty
        constant = True/False - pad regressors with 1s?

    OUTPUT
        betas = coefficients X n target variables
        resid = observations X n target variables
    """
    # Add 'constant' regressor
    if constant:
        ones = np.ones((regressors.shape[0],1))
        regressors = np.hstack((ones,regressors))
    X = regressors.copy()

    # construct regularization term
    LAMBDA = np.identity(X.shape[1])*alpha

    # Least squares minimization
    C_ss_inv = np.linalg.pinv(np.dot(X.T,X) + LAMBDA)
    
    betas = np.dot(C_ss_inv,np.dot(X.T,data))
    # Calculate residuals
    resid = data - (betas[0] + np.dot(X[:,1:],betas[1:]))

    return betas, resid

def computeGroupFCMappings_in_pc_space(inputtype):
    datadir = '/projects3/SRActFlow/data/results/GroupfMRI/RSA/LayerToLayerFC/'
    scount = 0
    for subj in subjNums:
        print 'Loading PCFC data for subject', scount+1, '/', len(subjNums)
        
        # Initialize first subj arrays arrays
        if scount == 0:
            fc_input2hidden, fc_hidden2motorresp, eig_input, eig_hidden = tools_group.loadSubjActFlowTaskFC(subj,inputtype,pc_space=True)
        else:
            tmp1, tmp2, tmp3, tmp4 = tools_group.loadSubjActFlowTaskFC(subj,inputtype,pc_space=True) 
            fc_input2hidden = fc_input2hidden + tmp1
            fc_hidden2motorresp = fc_hidden2motorresp + tmp2
            eig_input = eig_input + tmp3
            eig_hidden = eig_hidden + tmp4
        scount += 1
        

    # Compute average
    fc_input2hidden = np.divide(fc_input2hidden,float(len(subjNums)))
    fc_hidden2motorresp = np.divide(fc_hidden2motorresp,float(len(subjNums)))
    eig_input = np.divide(eig_input,float(len(subjNums)))
    eig_hidden = np.divide(eig_hidden,float(len(subjNums)))

    print 'Writing out to disk'
    if inputtype=='ori':
        # Store to h5f files
        h5f = h5py.File(datadir + 'ORIToHiddenLayer_PC_TaskFC_Group.h5','a')
        try:
            h5f.create_dataset('sourceToTargetMapping',data=fc_input2hidden)
            h5f.create_dataset('eigenvectors',data=eig_input)
        except:
            del h5f['sourceToTargetMapping'], h5f['eigenvectors']
            h5f.create_dataset('sourceToTargetMapping',data=fc_input2hidden)
            h5f.create_dataset('eigenvectors',data=eig_input)
        h5f.close()

    if inputtype=='color':
        # Store to h5f files
        h5f = h5py.File(datadir + 'COLORToHiddenLayer_PC_TaskFC_Group.h5','a')
        try:
            h5f.create_dataset('sourceToTargetMapping',data=fc_input2hidden)
            h5f.create_dataset('eigenvectors',data=eig_input)
        except:
            del h5f['sourceToTargetMapping'], h5f['eigenvectors']
            h5f.create_dataset('sourceToTargetMapping',data=fc_input2hidden)
            h5f.create_dataset('eigenvectors',data=eig_input)
        h5f.close()
                                        
    if inputtype=='pitch':
        # Store to h5f files
        h5f = h5py.File(datadir + 'PITCHToHiddenLayer_PC_TaskFC_Group.h5','a')
        try:
            h5f.create_dataset('sourceToTargetMapping',data=fc_input2hidden)
            h5f.create_dataset('eigenvectors',data=eig_input)
        except:
            del h5f['sourceToTargetMapping'], h5f['eigenvectors']
            h5f.create_dataset('sourceToTargetMapping',data=fc_input2hidden)
            h5f.create_dataset('eigenvectors',data=eig_input)
        h5f.close()
                                        
    if inputtype=='constant':
        # Store to h5f files
        h5f = h5py.File(datadir + 'CONSTANTToHiddenLayer_PC_TaskFC_Group.h5','a')
        try:
            h5f.create_dataset('sourceToTargetMapping',data=fc_input2hidden)
            h5f.create_dataset('eigenvectors',data=eig_input)
        except:
            del h5f['sourceToTargetMapping'], h5f['eigenvectors']
            h5f.create_dataset('sourceToTargetMapping',data=fc_input2hidden)
            h5f.create_dataset('eigenvectors',data=eig_input)
        h5f.close()
                                        
    if inputtype=='Logic' or inputtype=='Sensory' or inputtype=='Motor' or inputtype=='12':
        # Store to h5f files
        h5f = h5py.File(datadir + inputtype + 'RuleToHiddenLayer_PC_TaskFC_Group.h5','a')
        try:
            h5f.create_dataset('sourceToTargetMapping',data=fc_input2hidden)
            h5f.create_dataset('eigenvectors',data=eig_input)
        except:
            del h5f['sourceToTargetMapping'], h5f['eigenvectors']
            h5f.create_dataset('sourceToTargetMapping',data=fc_input2hidden)
            h5f.create_dataset('eigenvectors',data=eig_input)
        h5f.close()
                                        
    h5f = h5py.File(datadir + 'HiddenLayerToMotorResponsePC_TaskFC_Group.h5','a')
    try:
        h5f.create_dataset('sourceToTargetMapping',data=fc_hidden2motorresp)
        h5f.create_dataset('eigenvectors',data=eig_hidden)
    except:
        del h5f['sourceToTargetMapping'], h5f['eigenvectors']
        h5f.create_dataset('sourceToTargetMapping',data=fc_hidden2motorresp)
        h5f.create_dataset('eigenvectors',data=eig_hidden)
    h5f.close()

