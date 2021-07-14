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
import statsmodels.api as sm
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
from sklearn.linear_model import LinearRegression

projectdir = '/home/ti61/f_mc1689_1/SRActFlow/'
projectdir = '../../'

## General parameters/variables
nParcels = 360

glasserfile2 = projectdir + 'data/Q1-Q6_RelatedParcellation210.LR.CorticalAreas_dil_Colors.32k_fs_RL.dlabel.nii'
glasser2 = nib.load(glasserfile2).get_fdata()
glasser2 = np.squeeze(glasser2)


def loadInputActivity(subj,inputtype):
    x = tgp.loadTaskTiming(subj,'ALL')
    stimIndex = np.asarray(x['stimIndex'])
    ind = np.where(stimIndex==inputtype)[0]

    if inputtype!='64':
        datadir = projectdir + 'data/postProcessing/hcpPostProcCiric/'
        h5f = h5py.File(datadir + subj + '_glmOutput_data.h5','r')
        data = h5f['taskRegression/ALL_24pXaCompCorXVolterra_taskReg_betas_canonical'][:].copy()
        data = data[:,ind].copy()
        h5f.close()
    else:
        x = tgp.loadTaskTiming(subj,'64')
        stimIndex = np.asarray(x['stimIndex'])
        ind = np.where(stimIndex=='context')[0]
        datadir = projectdir + 'data/postProcessing/hcpPostProcCiric/'
        h5f = h5py.File(datadir + subj + '_glmoutput_data_64contextGLM.h5','r')
        data = h5f['taskRegression/64_24pXaCompCorXVolterra_taskReg_betas_canonical'][:].copy()
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
    
    datadir = projectdir + 'data/postProcessing/hcpPostProcCiric/'
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
    
    datadir = projectdir + 'data/postProcessing/hcpPostProcCiric/'
    h5f = h5py.File(datadir + subj + '_glmOutput_data.h5','r')
    data = h5f['taskRegression/ALL_24pXaCompCorXVolterra_taskReg_betas_canonical'][:].copy()
    data = data[:,ind].copy()
    h5f.close()
    
    # Isolate hand responses
    hand_ind = hands[hand]
    tmpdat = np.zeros((data.shape[0],2))
    if hand=='Right':
        #tmpdat[:,0] = data[:,3] #rmid --  need to flip this once glm is re-run -- check the new reffunc
        tmpdat[:,0] = np.real(data[:,2]) 
        tmpdat[:,1] = np.real(data[:,3]) 
    elif hand=='Left':
        tmpdat[:,0] = np.real(data[:,0]) #lmid
        tmpdat[:,1] = np.real(data[:,1]) #lind
    data = tmpdat.copy()

    return data

def loadMotorResponsesOutputMask():
    h5f = h5py.File(projectdir + 'data/results/MAIN/MotorResponseBetas_OutputVertices.h5','r')
    data_rh = h5f['RH'][:].copy()
    data_lh = h5f['LH'][:].copy()

    return data_rh, data_lh

def loadRestActivity(subj,model='24pXaCompCorXVolterra',zscore=False):
    datadir = projectdir + 'data/postProcessing/hcpPostProcCiric/'
    h5f = h5py.File(datadir + subj + '_glmOutput_data.h5','r')
    data = h5f['Rest1/nuisanceReg_resid_24pXaCompCorXVolterra'][:].copy()
    h5f.close()

    data = np.real(data)
    if zscore:
        data = stats.zscore(data,axis=1)
    return data

def loadTaskResiduals(subj,model='24pXaCompCorXVolterra',zscore=False):
    datadir = projectdir + 'data/postProcessing/hcpPostProcCiric/'
    h5f = h5py.File(datadir + subj + '_glmOutput_data.h5','r')
    data = h5f['taskRegression/ALL_24pXaCompCorXVolterra_taskReg_resid_FIR'][:].copy()
    h5f.close()

    data = np.real(data)
    if zscore:
        data = stats.zscore(data,axis=1)
    return data

def conditionDecodings(data, rois, ncvs=100, effects=False, motorOutput=False,confusion=False, decoder='similarity', nproc=5):
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
    confusion_mats = []
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
        
        scores, rmatch, rmismatch, confusion_mat = randomSplitLOOBaselineCV(ncvs, svm_mat, labels, subjarray, 
                                                                            motorOutput=motorOutput, decoder=decoder, nproc=nproc)
        stats[roicount,:] = scores
        rmatches[roicount] = np.mean(rmatch)
        rmismatches[roicount] = np.mean(rmismatch)
        confusion_mats.append(confusion_mat)
        roicount += 1
    
    if effects and confusion:
        return stats, rmatch, rmismatch, confusion_mats
    if effects and not confusion:
        return stats, rmatch, rmismatch
    if confusion and not effects:
        return stats, confusion_mats
    else:
        return stats

def randomSplitLOOBaselineCV(ncvs, svm_mat, labels, subjarray, motorOutput=False, permutation=False, decoder='similarity', nproc=5, featsel=False):
    """
    Runs cross validation for an across-subject SVM analysis
    """
    

    ntasks = len(np.unique(labels))
    nsamples = svm_mat.shape[0]
    nsubjs = nsamples/ntasks

    subjects = np.unique(subjarray)
    indices = np.arange(nsamples)
    
    numsubjs_perfold = 8
    if nsubjs%numsubjs_perfold!=0: 
        raise Exception("Error: Folds don't match number of subjects")
        
    nfolds = nsubjs/numsubjs_perfold
    subj_array_folds = subjarray.copy()
    
    inputs = [] 
    
    nfolds = int(nfolds)
    for fold in range(nfolds):
#        #test_subjs = np.random.choice(subj_array_folds,numsubjs_perfold,replace=False)
#        test_subjs = [subjects[fold]]
#        train_subjs_all = np.delete(subjects,test_subjs)
        ## run 8 fold CV
        test_subjs_ind = np.random.choice(np.arange(len(subj_array_folds)),numsubjs_perfold,replace=False)
        test_subjs = subj_array_folds[test_subjs_ind]
        # Training subjects are all subjects - those in test set
        train_subjs_all = np.delete(subjects,test_subjs)
        # For each fold, train on test subjects independently

        for test_subj in test_subjs:
            # Randomly sample half of train set subjects for each cv (CV bootstrapping)
            train_subjs = np.random.choice(train_subjs_all,
                                         int(len(train_subjs_all)),
                                         replace=True)
#            train_subjs = train_subjs_all

            train_ind = []
            for subj in train_subjs:
                train_ind.extend(np.where(subjarray==subj)[0])

            # Only obtain single test subject
            test_ind = np.where(subjarray==test_subj)[0]
            
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
            
            trainlabels = labels[train_ind].copy()
            testlabels = labels[test_ind].copy()


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
                #trainset_downsampled = trainset[:,h0]
                # Downsample test set into original vertices
                testset_downsampled = np.zeros((testset.shape[0],2))
                testset_downsampled[:,0] = np.nanmean(testset[:,feat1_mask],axis=1)
                testset_downsampled[:,1] = np.nanmean(testset[:,feat2_mask],axis=1)
                #testset_downsampled = testset[:,h0]

                ## permutation
                if permutation:
                    np.random.shuffle(trainlabels)
                
                if np.sum(feat1_mask)==0 or np.sum(feat2_mask==0):
                    inputs.append((trainset,testset,trainlabels,testlabels,decoder))
                else:
                    inputs.append((trainset_downsampled,testset_downsampled,trainlabels,testlabels,decoder))
            
            elif featsel:
                #### Revision addition - select for vertices based on FDR-corrected p<0.05 for noncircular accuracy decoding of motor output
                unique_labels = np.unique(labels)
                feat1_labs = np.where(trainlabels==unique_labels[0])[0]
                feat2_labs = np.where(trainlabels==unique_labels[1])[0]
                # Perform t-test
                t, p = stats.ttest_rel(trainset[feat1_labs,:],trainset[feat2_labs,:],axis=0)
                h0, qs = mc.fdrcorrection0(p)
                # Construct feature masks
                feat_mask = h0
                # use fdr-corrected vertices for feature selection 
                trainset = trainset[:,feat_mask]
                testset = testset[:,feat_mask]

                # if permutation
                if permutation:
                    np.random.shuffle(trainlabels)

                inputs.append((trainset,testset,trainlabels,testlabels,decoder))
            else:
                ## permutation
                if permutation:
                    np.random.shuffle(trainlabels)
#                trainlabels = labels[train_ind]
#                testlabels = labels[test_ind]
#                f, p = f_classif(trainset,trainlabels)
#                thresh = 0.1
#                feat_mask = p < thresh
#                inputs.append((trainset[:,feat_mask],testset[:,feat_mask],labels[train_ind],labels[test_ind]))         
            
                inputs.append((trainset,testset,trainlabels,testlabels,decoder))

            
        
        subj_array_folds = np.delete(subj_array_folds,test_subjs)
    
    #print('trainset.shape:', trainset.shape)
    #print('testset.shape:', testset.shape)
    #print('trainlabels:', inputs[0][2])
    #print('testlabels:', inputs[0][3])
    #print('inputs[0]:', len(inputs[0]))
    pool = mp.Pool(processes=nproc)
    scores = pool.starmap_async(_decoding,inputs).get()
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
    confusion_mat = []
    for score in scores:
        acc.extend(score[0])
        r_match.append(score[1])
        r_mismatch.append(score[2])
        confusion_mat.append(score[3])
       
    confusion_mat = np.asarray(confusion_mat)
    confusion_mat = np.mean(confusion_mat,axis=0)

    return acc, r_match, r_mismatch, confusion_mat

def _decoding(trainset,testset,trainlabels,testlabels,decoder='similarity',):
    unique_cond = np.unique(trainlabels)
    confusion_mat = np.zeros((len(unique_cond),len(unique_cond)))
    r_match = []
    r_mismatch = []
    acc = []
    i = 0
    for cond1 in unique_cond:
        mismatches = []
        prototype_ind = np.where(trainlabels==cond1)[0]
        prototype = np.mean(trainset[prototype_ind,:],axis=0)
        j = 0
        for cond2 in unique_cond:
            test_ind = np.where(testlabels==cond2)[0]
            test = np.mean(testset[test_ind,:],axis=0)
            if cond1 == cond2: 
                #correct = np.arctanh(stats.pearsonr(prototype,test)[0])
                correct = stats.pearsonr(prototype,test)[0]
                #correct = stats.spearmanr(prototype,test)[0]
                #correct = np.linalg.norm(prototype-test) 
                r_match.append(correct)
            else:
                #r = np.arctanh(stats.pearsonr(prototype,test)[0])
                r = stats.pearsonr(prototype,test)[0]
                #r = stats.spearmanr(prototype,test)[0]
                #r = np.linalg.norm(prototype-test) 
                mismatches.append(r)
                r_mismatch.append(r)
            j += 1
        
        
        if correct > np.max(mismatches): 
            acc.append(1.0)
            confusion_mat[i,i] = confusion_mat[i,i] + 1
        else:
            acc.append(0.0)
            prediction = np.where(mismatches==np.max(mismatches))[0][0]
            # mismatches only has n-1 elements, so need to find the actual class prediction
            prediction_adj = np.delete(np.arange(len(unique_cond)),i)[prediction]
            confusion_mat[i,prediction_adj] = confusion_mat[i,prediction_adj] + 1


        i += 1
       
    # if decoder is not similarity, re-do with a svm or logistic
    if decoder!='similarity': 
        if decoder=='logistic':
            clf = sklearn.linear_model.LogisticRegression(solver='liblinear',multi_class='ovr')
        elif decoder=='svm':
            clf = svm.SVC(kernel='linear')
        else:
            raise Exception("Not a valid decoder!")

        clf.fit(trainset,trainlabels)
        predictions = clf.predict(testset)
        acc = predictions==testlabels
        score = np.mean(acc)

    r_match = np.mean(r_match)
    r_mismatch = np.mean(r_mismatch)

    return acc, r_match, r_mismatch, confusion_mat

def actflowDecodings(data, actflow_data, effects=False, featsel=True, ncvs=1, permutation=False, confusion=False, nproc=5):
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

    scores, rmatch, rmismatch, confusion_mat = actflowRandomSplitLOOBaselineCV(ncvs, svm_mat, actflow_svm_mat, labels, subjarray, featsel=featsel, permutation=permutation, nproc=nproc)
#     stats = np.mean(scores)
    stats = scores 

    if effects and confusion:
        return stats, rmatch, rmismatch, confusion_mat
    if effects and not confusion:
        return stats, rmatch, rmismatch
    if confusion and not effects:
        return stats, confusion_mat
    else:
        return stats

def actflowRandomSplitLOOBaselineCV(ncvs, svm_mat, actflow_svm_mat, labels, subjarray, featsel=True, permutation=False,nproc=5):
    """
    Runs cross validation for an across-subject SVM analysis
    """
    
    ntasks = len(np.unique(labels))
    nsamples = svm_mat.shape[0]
    nsubjs = nsamples/ntasks

    subjects = np.unique(subjarray)
    indices = np.arange(nsamples)
    
    numsubjs_perfold = 8
    if nsubjs%numsubjs_perfold!=0: 
        raise Exception("Error: Folds don't match number of subjects")
        
    nfolds = nsubjs/numsubjs_perfold
    subj_array_folds = subjarray.copy()
    
    inputs = [] 
    
    nfolds = int(nfolds)
    for fold in range(nfolds):
        #test_subjs = np.random.choice(subj_array_folds,numsubjs_perfold,replace=False)
        #test_subjs = [subjects[fold]]
        # subj array folds are all subjects that have not yet been tested on
        test_subjs_ind = np.random.choice(np.arange(len(subj_array_folds)),numsubjs_perfold,replace=False)
        test_subjs = subj_array_folds[test_subjs_ind]
        # Training subjects are all subjects - those in test set
        train_subjs_all = np.delete(subjects,test_subjs)
        # For each fold, train on test subjects independently
        for test_subj in test_subjs:
            # Randomly sample half of train set subjects for each cv (CV bootstrapping)
            train_subjs = np.random.choice(train_subjs_all,
                                         int(len(train_subjs_all)),
                                         replace=True)
#            train_subjs = train_subjs_all

            train_ind = []
            for subj in train_subjs:
                train_ind.extend(np.where(subjarray==subj)[0])

            # Only obtain single test subject
            test_ind = np.where(subjarray==test_subj)[0]
            #test_ind = []
            #for subj in test_subjs:
            #    test_ind.extend(np.where(subjarray==subj)[0])
            
            train_ind = np.asarray(train_ind)
            test_ind = np.asarray(test_ind)


            ## Predicted-to-actual
            trainset = actflow_svm_mat[train_ind,:]
            testset = svm_mat[test_ind,:]

            ## Actual-to-predicted
#            trainset = svm_mat[train_ind,:]
#            testset = actflow_svm_mat[test_ind,:]
#            orig_training = svm_mat[train_ind,:]

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
            ## permutation
            if permutation:
                np.random.shuffle(trainlabels)
            ##
            unique_labels = np.unique(labels)
            #feat1_labs = np.where(trainlabels==0)[0]
            #feat2_labs = np.where(trainlabels==1)[0]
            ## Perform t-test
            #t, p = stats.ttest_rel(orig_training[feat1_labs,:],orig_training[feat2_labs,:],axis=0)
            ##t, p = stats.ttest_rel(trainset[feat1_labs,:],trainset[feat2_labs,:],axis=0)
            #h0, qs = mc.fdrcorrection0(p)

            F, p = f_classif(trainset,trainlabels)

            ### BEGIN REGULAR FEATURE SELECTION ###
            if featsel:
                thresh = 0.05
                feat_mask = np.where(p < thresh)[0]
                feat_mask = np.intersect1d(feat_mask,np.where(np.isnan(trainset[0,:])==False)[0]) # make sure no bad values are included
                inputs.append((trainset[:,feat_mask],testset[:,feat_mask],trainlabels,testlabels))         
                ### END REGULAR FEATURE SELECTION ###

#                ### BEGIN DIGIT REPRESENTATION FEATURE SELECTION ###
#                # Construct feature masks
#                feat1_mask = np.multiply(t<0,qs<0.05)
#                feat2_mask = np.multiply(t>0,qs<0.05)
#                #feat1_mask = t>0
#                #feat2_mask = t<0
#                
#                # Downsample training set into original vertices into 2 ROI signals
#                trainset_downsampled = np.zeros((trainset.shape[0],2))
#                trainset_downsampled[:,0] = np.nanmean(trainset[:,feat1_mask],axis=1)
#                trainset_downsampled[:,1] = np.nanmean(trainset[:,feat2_mask],axis=1)
#                # Downsample test set into original vertices
#                testset_downsampled = np.zeros((testset.shape[0],2))
#                testset_downsampled[:,0] = np.nanmean(testset[:,feat1_mask],axis=1)
#                testset_downsampled[:,1] = np.nanmean(testset[:,feat2_mask],axis=1)
#                if np.nansum(feat1_mask)==0 or np.nansum(feat2_mask)==0:
#                    print 'not running feature selection'
#                    inputs.append((trainset,testset,labels[train_ind],labels[test_ind]))
#                else:
#                    inputs.append((trainset_downsampled,testset_downsampled,labels[train_ind],labels[test_ind]))
#                ### END DIGIT REPRESENTATION FEATURE SELECTION ###
            else:
                inputs.append((trainset,testset,trainlabels,testlabels))         
    
        # Subj array folds are all subjects that have not yet been tested independently
        subj_array_folds = np.delete(subj_array_folds,test_subjs_ind)
        #subj_array_folds = np.delete(subj_array_folds,test_subjs)
        
    pool = mp.Pool(processes=nproc)
    scores = pool.starmap_async(_decoding,inputs).get()
    pool.close()
    pool.join()

    acc = []
    r_match = []
    r_mismatch = []
    confusion_mat = []
    for score in scores:
        acc.extend(score[0])
        r_match.append(score[1])
        r_mismatch.append(score[2])
        confusion_mat.append(score[3])
    
    confusion_mat = np.asarray(confusion_mat)
    confusion_mat = np.mean(confusion_mat,axis=0)

    return acc, r_match, r_mismatch, confusion_mat

def loadGroupActFlowFC(fcmapping,fcdir):
    if fcmapping in ['VERTICAL','RED','HIGH','CONSTANT']:
        h5f = h5py.File(fcdir + fcmapping + 'ToHiddenLayer_FC_Group.h5','r')
        eigs = h5f['eigenvectors'][:].copy()
        fc = h5f['sourceToTargetMapping'][:].copy()
        h5f.close()

    if fcmapping=='12' or fcmapping=='fpn' or fcmapping=='nounimodal':
        h5f = h5py.File(fcdir + fcmapping + 'RuleToHiddenLayer_FC_Group.h5','r')
        eigs = h5f['eigenvectors'][:].copy()
        fc = h5f['sourceToTargetMapping'][:].copy()
        h5f.close()

    if fcmapping=='hidden2out':
        h5f = h5py.File(fcdir + 'HiddenLayerToOutput_FC_Group.h5','r')
        eigs = h5f['eigenvectors'][:].copy()
        fc = h5f['sourceToTargetMapping'][:].copy()
        h5f.close()

    return np.real(fc), np.real(eigs)

def loadSubjActFlowFC(subj,inputtype,fcdir):
    if inputtype in ['VERTICAL','RED','HIGH','CONSTANT']:
        h5f = h5py.File(fcdir + inputtype + 'ToHiddenLayer_FC_subj' + subj + '.h5','r')
        fcmapping = h5f['sourceToTargetMapping'][:].copy()
        eigenvectors_input = h5f['eigenvectors'][:].copy()
        h5f.close()

        fc_input2hidden = fcmapping

    if inputtype=='12' or inputtype=='fpn' or inputtype=='nounimodal':
        h5f = h5py.File(fcdir + inputtype + 'RuleToHiddenLayer_FC_subj' + subj + '.h5','r')
        fcmapping = h5f['sourceToTargetMapping'][:].copy()
        eigenvectors_input = h5f['eigenvectors'][:].copy()
        h5f.close()

        fc_input2hidden = fcmapping

    h5f = h5py.File(fcdir + 'HiddenLayerToOutput_FC_subj' + subj + '.h5','r')
    fcmapping = h5f['sourceToTargetMapping'][:].copy()
    eigenvectors_hidden = h5f['eigenvectors'][:].copy()
    h5f.close()

    fc_hidden2motorresp = fcmapping


    return fc_input2hidden, fc_hidden2motorresp, eigenvectors_input, eigenvectors_hidden

def loadSubjActFlowFC_NoHidden(subj,inputtype,fcdir):
    """Load direct input 2 output FC mappings -- control analysis"""
    
    if inputtype in ['VERTICAL','RED','HIGH','CONSTANT']:
        h5f = h5py.File(fcdir + inputtype + 'ToOutputLayer_FC_subj' + subj + '.h5','r')
        fcmapping = h5f['sourceToTargetMapping'][:].copy()
        eigenvectors_input = h5f['eigenvectors'][:].copy()
        h5f.close()
        fc_input2output = fcmapping

    if inputtype=='12' or inputtype=='fpn' or inputtype=='nounimodal':
        h5f = h5py.File(fcdir + '12RuleToOutputLayer_FC_subj' + subj + '.h5','r')
        fcmapping = h5f['sourceToTargetMapping'][:].copy()
        eigenvectors_input = h5f['eigenvectors'][:].copy()
        h5f.close()
        fc_input2output = fcmapping

    return fc_input2output, eigenvectors_input

def loadGroupActFlowFC_NoHidden(inputtype,fcdir):
    """Load direct input 2 output FC mappings -- control analysis"""

    if inputtype in ['VERTICAL','RED','HIGH','CONSTANT']:
        h5f = h5py.File(fcdir + inputtype + 'ToOutputLayer_FC_Group.h5','r')
        fcmapping = h5f['sourceToTargetMapping'][:].copy()
        eigenvectors_input = h5f['eigenvectors'][:].copy()
        h5f.close()
        fc_input2output = fcmapping

    if inputtype=='12' or inputtype=='fpn' or inputtype=='nounimodal':
        h5f = h5py.File(fcdir + '12RuleToOutputLayer_FC_Group.h5','r')
        fcmapping = h5f['sourceToTargetMapping'][:].copy()
        eigenvectors_input = h5f['eigenvectors'][:].copy()
        h5f.close()
        fc_input2output = fcmapping

    return fc_input2output, eigenvectors_input

def mapBackToSurface(array,filename):
    """
    array can either be 360 array or ~59k array. If 360, will automatically map back to ~59k
    """
    #### Map back to surface
    if array.shape[0]==360:
        out_array = np.zeros((glasser2.shape[0],3))

        roicount = 0
        for roi in range(360):
            for col in range(array.shape[1]):
                vertex_ind = np.where(glasser2==roi+1)[0]
                out_array[vertex_ind,col] = array[roicount,col]

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


def multregconn(activity_matrix, target_ts=None, parcelstoexclude_bytarget=None, conn_mask=None):
    """
    activity_matrix:    Activity matrix should be nodes X time
    target_ts:             Optional, used when only a single target time series (returns 1 X nnodes matrix)
    parcelstoexclude_bytarget: Optional. A dictionary of lists, each listing parcels to exclude for each target parcel (e.g., to reduce potential circularity by removing parcels near the target parcel). Note: This is not used if target_ts is set.
    conn_mask: Optional. Specifies a mask to exclude some connections from being fit (setting them to 0). Consists of a matrix of 1s and 0s, where 1s indicate a connection and 0s indicate no connection. If target_ts=None then it is a N X N matrix (where N=number of nodes), otherwise it is N X 1.
    Output: connectivity_mat, formatted targets X sources
    """

    nnodes = activity_matrix.shape[0]
    timepoints = activity_matrix.shape[1]
    if nnodes > timepoints:
        print('activity_matrix shape: ',np.shape(activity_matrix))
        raise Exception('More nodes (regressors) than timepoints! Use regularized regression')

    if target_ts is None:
        connectivity_mat = np.zeros((nnodes,nnodes))
        for targetnode in range(nnodes):
            othernodes = list(range(nnodes))
            #Remove parcelstoexclude_bytarget parcels (if flagged); parcelstoexclude_bytarget is by index (not parcel value)
            if parcelstoexclude_bytarget is not None:
                parcelstoexclude_thisnode=parcelstoexclude_bytarget[targetnode].tolist()
                parcelstoexclude_thisnode.append(targetnode) # Remove target node from 'other nodes'
                othernodes = list(set(othernodes).difference(set(parcelstoexclude_thisnode)))
            elif conn_mask is not None:
                othernodes = list(set(othernodes).difference(set((conn_mask[targetnode,:]<1).nonzero()[0])))
            else:
                othernodes.remove(targetnode) # Remove target node from 'other nodes'
            X = activity_matrix[othernodes,:].T
            y = activity_matrix[targetnode,:]
            #Note: LinearRegression fits intercept by default (intercept beta not included in coef_ output)
            if len(othernodes)>0:
                reg = LinearRegression().fit(X, y)
                connectivity_mat[targetnode,othernodes]=reg.coef_
            # run multiple regression, and add constant
            #beta_fc,resids = regression.regression(y,X,alpha=0, constant=True) # increase alpha if want to apply a ridge penalty
            #connectivity_mat[targetnode,othernodes] = beta_fc[1:] # exclude 1st coef; first coef is beta_0 (or mean)
    else:
        #Computing values for a single target node
        connectivity_mat = np.zeros((nnodes,1))
        X = activity_matrix.T
        y = target_ts
        #Note: LinearRegression fits intercept by default (intercept beta not included in coef_ output)
        reg = LinearRegression().fit(X, y)
        connectivity_mat=reg.coef_
        # run multiple regression, and add constant
        #beta_fc,resids = regression.regression(y,X,alpha=0, constant=True) # increase alpha if want to apply a ridge penalty
        #connectivity_mat = beta_fc[1:] # exclude 1st coef; first coef is beta_0 (or mean)

    return connectivity_mat 



##### OLD
def compositionalActflowDecodings(data, nov_actflow_data, prac_actflow_data, effects=False, featsel=True, ncvs=1, nproc=5):
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
    nov_svm_mat = np.zeros((nsamples,nfeatures))
    prc_svm_mat = np.zeros((nsamples,nfeatures))
    samplecount = 0
    scount = 0
    for subj in range(nSubjs):
        origdata = data[:,:,scount]
        nov_data = nov_actflow_data[:,:,scount]
        prc_data = prac_actflow_data[:,:,scount]
        svm_mat[samplecount:(samplecount+ncond),:] = origdata.T
        nov_svm_mat[samplecount:(samplecount+ncond),:] = nov_data.T
        prc_svm_mat[samplecount:(samplecount+ncond),:] = prc_data.T

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

    scores, rmatch, rmismatch = compositionalActflowRandomSplitLOOBaselineCV(ncvs, svm_mat, nov_svm_mat, prc_svm_mat, labels, subjarray, featsel=featsel, nproc=nproc)
#     stats = np.mean(scores)
    stats = scores 
    if effects: 
        return stats, rmatch,rmismatch
    else:
        return stats

def compositionalActflowRandomSplitLOOBaselineCV(ncvs, svm_mat, nov_svm_mat, prc_svm_mat, labels, subjarray, featsel=True, nproc=5):
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

            trainset = prc_svm_mat[train_ind,:]
            testset = nov_svm_mat[test_ind,:]
            orig_training = svm_mat[train_ind,:]

            ## Normalize trainset and testset
            trainmean = np.mean(prc_svm_mat[train_ind,:],axis=0)
            trainmean.shape = (1,len(trainmean))
            trainstd = np.std(prc_svm_mat[train_ind,:],axis=0)
            trainstd.shape = (1,len(trainstd))
            #
            ## Normalize trainset and testset
            testmean = np.mean(nov_svm_mat[train_ind,:],axis=0)
            testmean.shape = (1,len(testmean))
            teststd = np.std(nov_svm_mat[train_ind,:],axis=0)
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
            t, p = stats.ttest_rel(orig_training[feat1_labs,:],orig_training[feat2_labs,:],axis=0)
            #t, p = stats.ttest_rel(trainset[feat1_labs,:],trainset[feat2_labs,:],axis=0)
            h0, qs = mc.fdrcorrection0(p)

            ### BEGIN REGULAR FEATURE SELECTION ###
            if featsel:
                thresh = 0.05
                feat_mask = np.where(qs < thresh)[0]
                feat_mask = np.intersect1d(feat_mask,np.where(np.isnan(trainset[0,:])==False)[0]) # make sure no bad values are included
                inputs.append((trainset[:,feat_mask],testset[:,feat_mask],labels[train_ind],labels[test_ind]))         
                ### END REGULAR FEATURE SELECTION ###

#                ### BEGIN DIGIT REPRESENTATION FEATURE SELECTION ###
#                # Construct feature masks
#                feat1_mask = np.multiply(t<0,qs<0.05)
#                feat2_mask = np.multiply(t>0,qs<0.05)
#                #feat1_mask = t>0
#                #feat2_mask = t<0
#                
#                # Downsample training set into original vertices into 2 ROI signals
#                trainset_downsampled = np.zeros((trainset.shape[0],2))
#                trainset_downsampled[:,0] = np.nanmean(trainset[:,feat1_mask],axis=1)
#                trainset_downsampled[:,1] = np.nanmean(trainset[:,feat2_mask],axis=1)
#                # Downsample test set into original vertices
#                testset_downsampled = np.zeros((testset.shape[0],2))
#                testset_downsampled[:,0] = np.nanmean(testset[:,feat1_mask],axis=1)
#                testset_downsampled[:,1] = np.nanmean(testset[:,feat2_mask],axis=1)
#                if np.nansum(feat1_mask)==0 or np.nansum(feat2_mask)==0:
#                    print 'not running feature selection'
#                    inputs.append((trainset,testset,labels[train_ind],labels[test_ind]))
#                else:
#                    inputs.append((trainset_downsampled,testset_downsampled,labels[train_ind],labels[test_ind]))
#                ### END DIGIT REPRESENTATION FEATURE SELECTION ###
            else:
                inputs.append((trainset,testset,labels[train_ind],labels[test_ind]))         
    
        subj_array_folds = np.delete(subj_array_folds,test_subjs)
        
    pool = mp.Pool(processes=nproc)
    scores = pool.starmap_async(_decoding,inputs).get()
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

