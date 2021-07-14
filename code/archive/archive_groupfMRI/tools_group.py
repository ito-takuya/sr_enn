# Taku Ito
# 2/22/2019
# General function modules for SRActFlow
# For group-level/cross-subject analyses

import numpy as np
import os
import multiprocessing as mp
import scipy.stats as stats
import nibabel as nib
import os
os.environ['OMP_NUM_THREADS'] = str(1)
#import statsmodels.api as sm
import sklearn.svm as svm
import statsmodels.sandbox.stats.multicomp as mc
import sklearn
from sklearn.feature_selection import f_classif
import h5py
os.sys.path.append('glmScripts/')
import taskGLMPipeline_v2 as tgp
import sys
sys.path.append('utils/')
import loadExperimentalData as led

basedir = '/projects3/SRActFlow/'

# Using final partition
networkdef = np.loadtxt('/projects3/NetworkDiversity/data/network_partition.txt')
networkorder = np.asarray(sorted(range(len(networkdef)), key=lambda k: networkdef[k]))
networkorder.shape = (len(networkorder),1)
# network mappings for final partition set
networkmappings = {'fpn':7, 'vis1':1, 'vis2':2, 'smn':3, 'aud':8, 'lan':6, 'dan':5, 'con':4, 'dmn':9, 
                           'pmulti':10, 'none1':11, 'none2':12}
networks = networkmappings.keys()

## General parameters/variables
nParcels = 360

glasserfile2 = '/projects/AnalysisTools/ParcelsGlasser2016/Q1-Q6_RelatedParcellation210.LR.CorticalAreas_dil_Colors.32k_fs_RL.dlabel.nii'
glasser2 = nib.load(glasserfile2).get_data()
glasser2 = np.squeeze(glasser2)


def loadInputActivity(subj,inputtype):
    x = tgp.loadTaskTiming(subj,'ALL')
    stimIndex = np.asarray(x['stimIndex'])
    ind = np.where(stimIndex==inputtype)[0]

    datadir = basedir + 'data/postProcessing/hcpPostProcCiric/'
    h5f = h5py.File(datadir + subj + '_glmOutput_data.h5','r')
    data = h5f['taskRegression/ALL_24pXaCompCorXVolterra_taskReg_betas_canonical'][:].copy()
    data = data[:,ind].copy()
    h5f.close()
    return data

def loadRuleEncoding(subj,rule='Logic'):
    """
    Loads average activity across trials (non-beta series)
    """
    
    x = tgp.loadTaskTiming(subj,'ALL')
    stimIndex = np.asarray(x['stimIndex'])
    ind = np.where(stimIndex==rule)[0]
    
    datadir = basedir + 'data/postProcessing/hcpPostProcCiric/'
    h5f = h5py.File(datadir + subj + '_glmOutput_data.h5','r')
    data = h5f['taskRegression/ALL_24pXaCompCorXVolterra_taskReg_betas_canonical'][:].copy()
    data = data[:,ind].copy()
    h5f.close()
    
    return data

def loadMotorResponses(subj,hand='Right'):
    """
    Loads average activity across trials (non-beta series)
    """
    
    hands = {'Left':[0,1],'Right':[2,3]}

    x = tgp.loadTaskTiming(subj,'ALL')
    stimIndex = np.asarray(x['stimIndex'])
    ind = np.where(stimIndex=='motorResponse')[0]
    
    datadir = basedir + 'data/postProcessing/hcpPostProcCiric/'
    h5f = h5py.File(datadir + subj + '_glmOutput_data.h5','r')
    data = h5f['taskRegression/ALL_24pXaCompCorXVolterra_taskReg_betas_canonical'][:].copy()
    data = data[:,ind].copy()
    h5f.close()
    
    # Isolate hand responses
    hand_ind = hands[hand]
    tmpdat = np.zeros((data.shape[0],2))
    if hand=='Right':
        #tmpdat[:,0] = data[:,3] #rmid --  need to flip this once glm is re-run -- check the new reffunc
        tmpdat[:,0] = data[:,2] 
        tmpdat[:,1] = data[:,3] 
    elif hand=='Left':
        tmpdat[:,0] = data[:,0] #lmid
        tmpdat[:,1] = data[:,1] #lind
    data = tmpdat.copy()
    
    return data

def loadRestActivity(subj,model='24pXaCompCorXVolterra',zscore=False):
    datadir = basedir + 'data/postProcessing/hcpPostProcCiric/'
    h5f = h5py.File(datadir + subj + '_glmOutput_data.h5','r')
    data = h5f['Rest1/nuisanceReg_resid_24pXaCompCorXVolterra'][:].copy()
    h5f.close()

    if zscore:
        data = stats.zscore(data,axis=1)
    return data

def loadMotorRuleToMotorOutputFC(subj):
    fcdir = '/projects3/SRActFlow/data/results/GroupfMRI/LayerToLayerFC/'
    filename = fcdir + 'MotorRuleToMotorResponseFC_subj' + subj + '.h5'
    h5f = h5py.File(filename,'r')
    fcmapping = h5f['sourceToTargetMapping'][:].copy()
    eigenvector = h5f['eigenvectors'][:].copy()
    h5f.close()
    # Map back to brain space
    fcmapping = np.dot(fcmapping.T,eigenvector).T
    
    return fcmapping 

def loadSRToMotorRuleFC(sr_rule,subj):
    fcdir = '/projects3/SRActFlow/data/results/GroupfMRI/LayerToLayerFC/'
    filename = fcdir + sr_rule + 'ToMotorDecisionFC_subj' + subj + '.h5'
    h5f = h5py.File(filename,'r')
    fcmapping = h5f['sourceToTargetMapping'][:].copy()
    eigenvector = h5f['eigenvectors'][:].copy()
    h5f.close()
    # Map back to brain space
    fcmapping = np.dot(fcmapping.T,eigenvector).T
    
    return fcmapping 

def loadInputToSRFC(inputtype,sr_rule,subj):
    fcdir = '/projects3/SRActFlow/data/results/GroupfMRI/LayerToLayerFC/'
    filename = fcdir + + inputtype + 'To' + sr_rule + 'FC_subj' + subj + '.h5'
    h5f = h5py.File(filename,'r')
    fcmapping = h5f['sourceToTargetMapping'][:].copy()
    eigenvector = h5f['eigenvectors'][:].copy()
    h5f.close()
    # Map back to brain space
    fcmapping = np.dot(fcmapping.T,eigenvector).T
    
    return fcmapping 

def loadPcaFCNoColliders(subj,roi):
    fcdir = '/projects3/SRActFlow/data/results/pcaFC/'
#     filename = fcdir + 'TargetParcel' + str(roi) + '_RidgeFC.h5'
    filename = fcdir + 'TargetParcel' + str(roi) + '_pcaFC_nozscore_noColliders.h5'
    h5f = h5py.File(filename,'r')
    fcmapping = h5f[subj]['sourceToTargetmapping'][:].copy()
    h5f.close()
    
    return fcmapping 

## Load masks
def loadMask(roi,dilated=True):
    maskdir = basedir + 'data/results/surfaceMasks/'
    if dilated:
        maskfile = maskdir + 'GlasserParcel' + str(roi) + '_dilated_10mm.dscalar.nii'
    else:
        maskfile = maskdir + 'GlasserParcel' + str(roi) + '.dscalar.nii'
    maskdata = np.squeeze(nib.load(maskfile).get_data())
    
    return maskdata
        
def conditionDecodings(data, rois, ncvs=100, effects=False, motorOutput=False, nproc=5):
    """
    Run an across-subject classification
    Decode responses on each hand separately from CPRO data
    Limit to ROIs within SMN network
    """
    
    ncond = data.shape[1] # two motor outputs
    nSubjs = data.shape[2]

    nsamples = nSubjs * ncond
    stats = np.zeros((len(rois),nsamples))
    rmatches = np.zeros((len(rois),))
    rmismatches = np.zeros((len(rois),))

    # Label array for supervised learning
    labels = np.tile(range(ncond),nSubjs)
    subjarray = np.repeat(range(nSubjs),ncond)

    # Run SVM classifications on network-level activation patterns across subjects
    roicount = 0
    for roi in rois:
        roi_ind = np.where(glasser2==roi+1)[0]
        nfeatures = len(roi_ind)
        roi_ind.shape = (len(roi_ind),1)       

        svm_mat = np.zeros((nsamples,roi_ind.shape[0]))
        samplecount = 0
        for scount in range(nSubjs):
            roidata = np.squeeze(data[roi_ind,:,scount])
            svm_mat[samplecount:(samplecount+ncond),:] = roidata.T

            samplecount += ncond

        # Spatially demean matrix across features
#        samplemean = np.mean(svm_mat,axis=1)
#        samplemean.shape = (len(samplemean),1)
#        svm_mat = svm_mat - samplemean
        
        scores, rmatch, rmismatch = randomSplitLOOBaselineCV(ncvs, svm_mat, labels, subjarray, motorOutput=motorOutput, nproc=nproc)
        stats[roicount,:] = scores
        rmatches[roicount] = np.mean(rmatch)
        rmismatches[roicount] = np.mean(rmismatch)
        roicount += 1
    
    if effects:
        return stats, rmatches, rmismatches
    else:
        return stats

def randomSplitLOOBaselineCV(ncvs, svm_mat, labels, subjarray, motorOutput=False, nproc=5):
    """
    Runs cross validation for an across-subject SVM analysis
    """
    
    ntasks = len(np.unique(labels))
    nsamples = svm_mat.shape[0]
    nsubjs = nsamples/ntasks

    subjects = np.unique(subjarray)
    indices = np.arange(nsamples)
    
    numsubjs_perfold = 1
    if nsubjs%numsubjs_perfold!=0: 
        raise Exception("Error: Folds don't match number of subjects")
        
    nfolds = nsubjs/numsubjs_perfold
    subj_array_folds = subjarray.copy()
    
    inputs = [] 
    
    for fold in range(nfolds):
        #test_subjs = np.random.choice(subj_array_folds,numsubjs_perfold,replace=False)
        test_subjs = [subjects[fold]]
        train_subjs_all = np.delete(subjects,test_subjs)
        for cv in range(ncvs):
            # Randomly sample half of train set subjects for each cv (CV bootstrapping)
            train_subjs = np.random.choice(train_subjs_all,
                                         int(np.floor(len(train_subjs_all)*(5.0))),
                                         replace=True)
#            train_subjs = train_subjs_all

            train_ind = []
            for subj in train_subjs:
                train_ind.extend(np.where(subjarray==subj)[0])

            test_ind = []
            for subj in test_subjs:
                test_ind.extend(np.where(subjarray==subj)[0])
            
            train_ind = np.asarray(train_ind)
            test_ind = np.asarray(test_ind)

            trainset = svm_mat[train_ind,:]
            testset = svm_mat[test_ind,:]

            # Normalize trainset and testset
            mean = np.mean(svm_mat[train_ind,:],axis=0)
            mean.shape = (1,len(mean))
            std = np.std(svm_mat[train_ind,:],axis=0)
            std.shape = (1,len(std))

            trainset = np.divide((trainset - mean),std)
            testset = np.divide((testset - mean),std)
            
            trainlabels = labels[train_ind]
            testlabels = labels[test_ind]

            if motorOutput:
                ## Feature selection and downsampling
                unique_labels = np.unique(labels)
                feat1_labs = np.where(trainlabels==unique_labels[0])[0]
                feat2_labs = np.where(trainlabels==unique_labels[1])[0]
                # Perform t-test
                t, p = stats.ttest_rel(trainset[feat1_labs,:],trainset[feat2_labs,:],axis=0)
                h0, qs = mc.fdrcorrection0(p)
    #             h0 = p<0.1
                # Construct feature masks
                feat1_mask = np.multiply(t>0,h0).astype(bool)
                feat2_mask = np.multiply(t<0,h0).astype(bool)
    #             feat1_mask = t>0
    #             feat2_mask = t<0
                # Downsample training set into original vertices into 2 ROI signals
                trainset_downsampled = np.zeros((trainset.shape[0],2))
                trainset_downsampled[:,0] = np.nanmean(trainset[:,feat1_mask],axis=1)
                trainset_downsampled[:,1] = np.nanmean(trainset[:,feat2_mask],axis=1)
                trainset_downsampled = trainset[:,h0]
                # Downsample test set into original vertices
                testset_downsampled = np.zeros((testset.shape[0],2))
                testset_downsampled[:,0] = np.nanmean(testset[:,feat1_mask],axis=1)
                testset_downsampled[:,1] = np.nanmean(testset[:,feat2_mask],axis=1)
                testset_downsampled = testset[:,h0]
                
                if np.sum(feat1_mask)==0 or np.sum(feat2_mask==0):
                    inputs.append((trainset,testset,trainlabels,testlabels))
                else:
                    inputs.append((trainset_downsampled,testset_downsampled,trainlabels,testlabels))
            else:
                trainlabels = labels[train_ind]
                testlabels = labels[test_ind]
                f, p = f_classif(trainset,trainlabels)
                thresh = 0.1
                feat_mask = p < thresh
                inputs.append((trainset[:,feat_mask],testset[:,feat_mask],labels[train_ind],labels[test_ind]))         
            
                #inputs.append((trainset,testset,trainlabels,testlabels))
            
        
        subj_array_folds = np.delete(subj_array_folds,test_subjs)
        
    pool = mp.Pool(processes=nproc)
    scores = pool.map_async(_decoding,inputs).get()
    pool.close()
    pool.join()
    
#     subj_acc = np.zeros((len(subjects),))
#     scount = 0
#     i = 0
#     for subj in subjects:
#         subjmean = []
#         for cv in range(ncvs):
#             subjmean.append(scores[i])
#             i += 1
        
#         subj_acc[scount] = np.mean(subjmean)
        
#         scount += 1

#     return subj_acc
    acc = []
    r_match = []
    r_mismatch = []
    for score in scores:
        acc.extend(score[0])
        r_match.append(score[1])
        r_mismatch.append(score[2])
        
    return acc, r_match, r_mismatch

def _decoding((trainset,testset,trainlabels,testlabels)):
    unique_cond = np.unique(trainlabels)
    rdm = np.zeros((len(unique_cond),len(unique_cond)))
    r_match = []
    r_mismatch = []
    acc = []
    for cond1 in unique_cond:
        mismatches = []
        prototype_ind = np.where(trainlabels==cond1)[0]
        prototype = np.mean(trainset[prototype_ind,:],axis=0)
        for cond2 in unique_cond:
            test_ind = np.where(testlabels==cond2)[0]
            test = np.mean(testset[test_ind,:],axis=0)
            if cond1 == cond2: 
                correct = np.arctanh(stats.pearsonr(prototype,test)[0])
                #correct = stats.spearmanr(prototype,test)[0]
                #correct = np.linalg.norm(prototype-test) 
                r_match.append(correct)
            else:
                r = np.arctanh(stats.pearsonr(prototype,test)[0])
                #r = stats.spearmanr(prototype,test)[0]
                #r = np.linalg.norm(prototype-test) 
                mismatches.append(r)
                r_mismatch.append(r)
        
        if correct > np.max(mismatches): 
            acc.append(1.0)
        else:
            acc.append(0.0)
    
#    clf = sklearn.linear_model.LogisticRegression()
#    clf = svm.SVC(C=1.0, kernel='linear')

#    clf.fit(trainset,trainlabels)
#    predictions = clf.predict(testset)
#    acc = predictions==testlabels
#    score = np.mean(acc)

    r_match = np.mean(r_match)
    r_mismatch = np.mean(r_mismatch)

    return acc, r_match, r_mismatch

def actflowDecodings(data, actflow_data, effects=False, ncvs=1, nproc=5):
    """
    Run an across-subject classification
    Decode responses on each hand separately from CPRO data
    """

    nSubjs = data.shape[2]
    stats = np.zeros((1,))
    
    ncond = data.shape[1]

    nsamples = nSubjs * ncond
    nfeatures = data.shape[0]

    # Label array for supervised learning
    labels = np.tile(range(ncond),nSubjs)
    subjarray = np.repeat(range(nSubjs),ncond)

    svm_mat = np.zeros((nsamples,nfeatures))
    actflow_svm_mat = np.zeros((nsamples,nfeatures))
    samplecount = 0
    scount = 0
    for subj in range(nSubjs):
        roidata = data[:,:,scount]
        actflow_roidata = actflow_data[:,:,scount]
        svm_mat[samplecount:(samplecount+ncond),:] = roidata.T
        actflow_svm_mat[samplecount:(samplecount+ncond),:] = actflow_roidata.T

        scount += 1
        samplecount += ncond

        # Spatially demean matrix across features
#        samplemean = np.mean(svm_mat,axis=1)
#        samplemean.shape = (len(samplemean),1)
#        svm_mat = svm_mat - samplemean
#
#        samplemean = np.mean(actflow_svm_mat,axis=1)
#        samplemean.shape = (len(samplemean),1)
#        actflow_svm_mat = actflow_svm_mat - samplemean

    scores, rmatch, rmismatch= actflowRandomSplitLOOBaselineCV(ncvs, svm_mat, actflow_svm_mat, labels, subjarray, nproc=nproc)
#     stats = np.mean(scores)
    stats = scores 
    if effects: 
        return stats, rmatch,rmismatch
    else:
        return stats

def actflowRandomSplitLOOBaselineCV(ncvs, svm_mat, actflow_svm_mat, labels, subjarray, nproc=5):
    """
    Runs cross validation for an across-subject SVM analysis
    """
    
    ntasks = len(np.unique(labels))
    nsamples = svm_mat.shape[0]
    nsubjs = nsamples/ntasks

    subjects = np.unique(subjarray)
    indices = np.arange(nsamples)
    
    numsubjs_perfold = 1
    if nsubjs%numsubjs_perfold!=0: 
        raise Exception("Error: Folds don't match number of subjects")
        
    nfolds = nsubjs/numsubjs_perfold
    subj_array_folds = subjarray.copy()
    
    inputs = [] 
    
    for fold in range(nfolds):
        #test_subjs = np.random.choice(subj_array_folds,numsubjs_perfold,replace=False)
        test_subjs = [subjects[fold]]
        train_subjs_all = np.delete(subjects,test_subjs)
        for cv in range(ncvs):
            # Randomly sample half of train set subjects for each cv (CV bootstrapping)
#            train_subjs = np.random.choice(train_subjs_all,
#                                         int(np.floor(len(train_subjs_all)*(4.0))),
#                                         replace=True)
            train_subjs = train_subjs_all

            train_ind = []
            for subj in train_subjs:
                train_ind.extend(np.where(subjarray==subj)[0])

            test_ind = []
            for subj in test_subjs:
                test_ind.extend(np.where(subjarray==subj)[0])
            
            train_ind = np.asarray(train_ind)
            test_ind = np.asarray(test_ind)

            trainset = actflow_svm_mat[train_ind,:]
            testset = svm_mat[test_ind,:]
            orig_training = svm_mat[train_ind,:]

            # Normalize trainset and testset
            trainmean = np.mean(actflow_svm_mat[train_ind,:],axis=0)
            trainmean.shape = (1,len(trainmean))
            trainstd = np.std(actflow_svm_mat[train_ind,:],axis=0)
            trainstd.shape = (1,len(trainstd))
            
            # Normalize trainset and testset
            testmean = np.mean(svm_mat[train_ind,:],axis=0)
            testmean.shape = (1,len(testmean))
            teststd = np.std(svm_mat[train_ind,:],axis=0)
            teststd.shape = (1,len(teststd))

            trainset = np.divide((trainset - trainmean),trainstd)
            testset = np.divide((testset - testmean),teststd)


             ######## FEATURE SELECTION & REDUCTION
             ## Feature selection and downsampling
            trainlabels = labels[train_ind]
            testlabels = labels[test_ind]
            unique_labels = np.unique(labels)
            feat1_labs = np.where(trainlabels==0)[0]
            feat2_labs = np.where(trainlabels==1)[0]
            # Perform t-test
            #t, p = stats.ttest_rel(orig_training[feat1_labs,:],orig_training[feat2_labs,:],axis=0)
            t, p = stats.ttest_rel(trainset[feat1_labs,:],trainset[feat2_labs,:],axis=0)
            h0, qs = mc.fdrcorrection0(p)
            thresh = 1.0
            feat_mask = np.where(p < thresh)[0]
            feat_mask = np.intersect1d(feat_mask,np.where(np.isnan(trainset[0,:])==False)[0]) # make sure no bad values are included
            inputs.append((trainset[:,feat_mask],testset[:,feat_mask],labels[train_ind],labels[test_ind]))         
            # Construct feature masks
#            feat1_mask = np.multiply(t<0,h0)
#            feat2_mask = np.multiply(t>0,h0)
#            feat1_mask = t>0
#            feat2_mask = t<0
#            # Downsample training set into original vertices into 2 ROI signals
#            trainset_downsampled = np.zeros((trainset.shape[0],2))
#            trainset_downsampled[:,0] = np.nanmean(trainset[:,feat1_mask],axis=1)
#            trainset_downsampled[:,1] = np.nanmean(trainset[:,feat2_mask],axis=1)
#            trainset_downsampled = trainset[:,h0]
#            # Downsample test set into original vertices
#            testset_downsampled = np.zeros((testset.shape[0],2))
#            testset_downsampled[:,0] = np.nanmean(testset[:,feat1_mask],axis=1)
#            testset_downsampled[:,1] = np.nanmean(testset[:,feat2_mask],axis=1)
#            testset_downsampled = testset[:,h0]
#              print 'feat1_mask', np.sum(feat1_mask), '| feat2_mask', np.sum(feat2_mask)

#            if np.sum(feat1_mask)==0 or np.sum(feat2_mask)==0:
#                print 'not running feature selection'
#                inputs.append((trainset,testset,labels[train_ind],labels[test_ind]))
#            else:
#                inputs.append((trainset_downsampled,testset_downsampled,labels[train_ind],labels[test_ind]))

            #inputs.append((trainset,testset,labels[train_ind],labels[test_ind]))         
    
        subj_array_folds = np.delete(subj_array_folds,test_subjs)
        
    pool = mp.Pool(processes=nproc)
    scores = pool.map_async(_decoding,inputs).get()
    pool.close()
    pool.join()

    acc = []
    r_match = []
    r_mismatch = []
    for score in scores:
        acc.extend(score[0])
        r_match.append(score[1])
        r_mismatch.append(score[2])
        
    return acc, r_match, r_mismatch

def loadGroupActFlowFC(inputtype):
    fcdir = '/projects3/SRActFlow/data/results/GroupfMRI/LayerToLayerFC/'
    if inputtype=='ori':
        h5f = h5py.File(fcdir + 'ORITosrVerticalFC_Group.h5','r')
        fc_input2sr = h5f['sourceToTargetMapping'][:].copy()
        h5f.close()

        h5f = h5py.File(fcdir + 'srVerticalToMotorDecisionFC_Group.h5','r')
        fc_sr2motorrule = h5f['sourceToTargetMapping'][:].copy()
        h5f.close()

    if inputtype=='color':
        h5f = h5py.File(fcdir + 'COLORTosrRedFC_Group.h5','r')
        fc_input2sr = h5f['sourceToTargetMapping'][:].copy()
        h5f.close()

        h5f = h5py.File(fcdir + 'srRedToMotorDecisionFC_Group.h5','r')
        fc_sr2motorrule = h5f['sourceToTargetMapping'][:].copy()
        h5f.close()
   
    if inputtype=='pitch':
        h5f = h5py.File(fcdir + 'PITCHTosrHighFC_Group.h5','r')
        fc_input2sr = h5f['sourceToTargetMapping'][:].copy()
        h5f.close()

        h5f = h5py.File(fcdir + 'srHighToMotorDecisionFC_Group.h5','r')
        fc_sr2motorrule = h5f['sourceToTargetMapping'][:].copy()
        h5f.close()
    
    if inputtype=='constant':
        h5f = h5py.File(fcdir + 'CONSTANTTosrConstantFC_Group.h5','r')
        fc_input2sr = h5f['sourceToTargetMapping'][:].copy()
        h5f.close()

        h5f = h5py.File(fcdir + 'srConstantToMotorDecisionFC_Group.h5','r')
        fc_sr2motorrule = h5f['sourceToTargetMapping'][:].copy()
        h5f.close()

    h5f = h5py.File(fcdir + 'MotorRuleToMotorResponseFC_Group.h5','r')
    fc_motorrule2motorresp= h5f['sourceToTargetMapping'][:].copy()
    h5f.close()

    return fc_input2sr, fc_sr2motorrule, fc_motorrule2motorresp

def loadSubjActFlowFC(subj,inputtype):
    fcdir = '/projects3/SRActFlow/data/results/GroupfMRI/LayerToLayerFC/'
    if inputtype=='ori':
        h5f = h5py.File(fcdir + 'ORITosrVerticalFC_subj' + subj + '.h5','r')
        fcmapping = h5f['sourceToTargetMapping'][:].copy()
        eigenvectors = h5f['eigenvectors'][:].copy()
        h5f.close()
        fc_input2sr = np.dot(fcmapping.T,eigenvectors).T

        h5f = h5py.File(fcdir + 'srVerticalToMotorDecisionFC_subj' + subj + '.h5','r')
        fcmapping = h5f['sourceToTargetMapping'][:].copy()
        eigenvectors = h5f['eigenvectors'][:].copy()
        h5f.close()
        fc_sr2motorrule = np.dot(fcmapping.T,eigenvectors).T

    if inputtype=='color':
        h5f = h5py.File(fcdir + 'COLORTosrRedFC_subj' + subj + '.h5','r')
        fcmapping = h5f['sourceToTargetMapping'][:].copy()
        eigenvectors = h5f['eigenvectors'][:].copy()
        h5f.close()
        fc_input2sr = np.dot(fcmapping.T,eigenvectors).T

        h5f = h5py.File(fcdir + 'srRedToMotorDecisionFC_subj' + subj + '.h5','r')
        fcmapping = h5f['sourceToTargetMapping'][:].copy()
        eigenvectors = h5f['eigenvectors'][:].copy()
        h5f.close()
        fc_sr2motorrule = np.dot(fcmapping.T,eigenvectors).T
   
    if inputtype=='pitch':
        h5f = h5py.File(fcdir + 'PITCHTosrHighFC_subj' + subj + '.h5','r')
        fcmapping = h5f['sourceToTargetMapping'][:].copy()
        eigenvectors = h5f['eigenvectors'][:].copy()
        h5f.close()
        fc_input2sr = np.dot(fcmapping.T,eigenvectors).T

        h5f = h5py.File(fcdir + 'srHighToMotorDecisionFC_subj' + subj + '.h5','r')
        fcmapping = h5f['sourceToTargetMapping'][:].copy()
        eigenvectors = h5f['eigenvectors'][:].copy()
        h5f.close()
        fc_sr2motorrule = np.dot(fcmapping.T,eigenvectors).T
    
    if inputtype=='constant':
        h5f = h5py.File(fcdir + 'CONSTANTTosrConstantFC_subj' + subj + '.h5','r')
        fcmapping = h5f['sourceToTargetMapping'][:].copy()
        eigenvectors = h5f['eigenvectors'][:].copy()
        h5f.close()
        fc_input2sr = np.dot(fcmapping.T,eigenvectors).T

        h5f = h5py.File(fcdir + 'srConstantToMotorDecisionFC_subj' + subj + '.h5','r')
        fcmapping = h5f['sourceToTargetMapping'][:].copy()
        eigenvectors = h5f['eigenvectors'][:].copy()
        h5f.close()
        fc_sr2motorrule = np.dot(fcmapping.T,eigenvectors).T

    h5f = h5py.File(fcdir + 'MotorRuleToMotorResponseFC_subj' + subj + '.h5','r')
    fcmapping = h5f['sourceToTargetMapping'][:].copy()
    eigenvectors = h5f['eigenvectors'][:].copy()
    h5f.close()
    fc_motorrule2motorresp = np.dot(fcmapping.T,eigenvectors).T

    return fc_input2sr, fc_sr2motorrule, fc_motorrule2motorresp


def mapBackToSurface(array,filename):
    """
    array can either be 360 array or ~59k array. If 360, will automatically map back to ~59k
    """
    #### Map back to surface
    if array.shape[0]==360:
        out_array = np.zeros((glasser2.shape[0],3))

        roicount = 0
        for roi in rois:
            for col in range(array.shape[1]):
                vertex_ind = np.where(glasser2==roi+1)[0]
                out_array[vertex_ind,0] = array[roicount,0]
                out_array[vertex_ind,1] = array[roicount,1]
                out_array[vertex_ind,2] = array[roicount,2]

            roicount += 1

    else:
        out_array = array

    #### 
    # Write file to csv and run wb_command
    np.savetxt(filename + '.csv', out_array,fmt='%s')
    wb_file = filename + '.dscalar.nii'
    wb_command = 'wb_command -cifti-convert -from-text ' + filename + '.csv ' + glasserfile2 + ' ' + wb_file + ' -reset-scalars'
    os.system(wb_command)
    os.remove(filename + '.csv')
