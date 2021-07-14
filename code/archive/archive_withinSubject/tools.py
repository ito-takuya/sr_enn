# Taku Ito
# 2/22/2019
# General function modules for SRActFlow

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
import taskGLMPipeline as tgp
import statsmodels.api as sm
import sys
sys.path.append('utils/')
import loadExperimentalData as led


basedir = '/projects3/SRActFlow/'


def loadMotorResponses(subj):
#     x = tgp.loadTaskTiming(subj,'betaSeries')
#     stimIndex = np.asarray(x['stimIndex'])
#     ind = np.where(stimIndex=='motorResponse')[0]
    datadir = basedir + 'data/postProcessing/hcpPostProcCiric/'
    h5f = h5py.File(datadir + subj + '_glmOutput_data.h5','r')
    data = h5f['taskRegression/betaSeries_24pXaCompCorXVolterra_taskReg_betas_canonical'][:].copy()
    # Probe activations are starting from index 128 (first 128 are encoding activations)
    data = data[:,128:].copy()
#     data = np.loadtxt(datadir + subj + '_motorResponse_taskBetas_Surface64k_GSR.csv',delimiter=',')
#     data = data[:,-4:]
    h5f.close()
    return data


def loadCrossTrialRuleEncoding(subj,rule='Logic'):
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

def loadCrossTrialMotorResponses(subj,hand='Right'):
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
        tmpdat[:,0] = data[:,3]
        tmpdat[:,1] = data[:,2]
    elif hand=='Left':
        tmpdat[:,0] = data[:,0]
        tmpdat[:,1] = data[:,1]
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


def loadPcaFCNoColliders(subj,roi):
    fcdir = '/projects3/SRActFlow/data/results/pcaFC/'
#     filename = fcdir + 'TargetParcel' + str(roi) + '_RidgeFC.h5'
    filename = fcdir + 'TargetParcel' + str(roi) + '_pcaFC_nozscore_noColliders.h5'
    h5f = h5py.File(filename,'r')
    fcmapping = h5f[subj]['sourceToTargetmapping'][:].copy()
    h5f.close()
    
    return fcmapping 

def loadFCandEig(subj,roi):
    fcdir = '/projects3/SRActFlow/data/results/pcaFC/'
#     filename = fcdir + 'TargetParcel' + str(roi) + '_RidgeFC.h5'
    filename = fcdir + 'TargetParcel' + str(roi) + '_pcaFC_nozscore.h5'
    h5f = h5py.File(filename,'r')
    fcmapping = h5f[subj]['sourceToTargetMapping'][:].copy()
    eigenvectors = h5f[subj]['eigenvectors'][:].copy()
    h5f.close()
    
    
    ## Remove collider coefficients
#     tmp = np.corrcoef(eigenvectors)
#     colliders = []
#     count = 0
#     for i in range(eigenvectors.shape[0]):
#         for j in range(eigenvectors.shape[0]):
#             if i==j: continue
#             if i<j: 
#                 # If it is potentially a collider variable, set the coefficient to 0
#                 if np.abs(tmp[i,j]) > 0.1:
#                     fcmapping[j,:] = 0
#                     colliders.append(j)
#                     count += 1
                    
#     print colliders
#     print float(count)/fcmapping.shape[0]
    return fcmapping, eigenvectors


## Load masks
def loadMask(roi,dilated=True):
    maskdir = basedir + 'data/results/surfaceMasks/'
    if dilated:
        maskfile = maskdir + 'GlasserParcel' + str(roi) + '_dilated_10mm.dscalar.nii'
    else:
        maskfile = maskdir + 'GlasserParcel' + str(roi) + '.dscalar.nii'
    maskdata = np.squeeze(nib.load(maskfile).get_data())
    
    return maskdata
        
        
def motorResponseDecodings((data, actflow, subj, hand, ncvs)):
    """
    Run a within-subject classification
    Assumes data is a space X feature matrix
    Decode responses on each hand separately from CPRO data
    """
    
    df_task = led.loadExperimentalData(subj) 
    # Motor responses are 'b (LMID), y (LIND), g (RIND), r (RMID)'
    motor_responses = df_task['MotorResponses'].values
    if hand=='left':
        fing1_ind = np.where(motor_responses=='b')[0] #lmid
        fing2_ind = np.where(motor_responses=='y')[0] #lind
    elif hand=='right':
        fing1_ind = np.where(motor_responses=='g')[0] #rind
        fing2_ind = np.where(motor_responses=='r')[0] #rmid
    
    fing1_nsamples = len(fing1_ind)
    fing2_nsamples = len(fing2_ind)
    
    labels = []
    labels.extend(np.repeat(0,fing1_nsamples))
    labels.extend(np.repeat(1,fing2_nsamples))
    
    # Find the minimum number of unique samples
    min_unique_samples = np.min([fing1_nsamples,fing2_nsamples])
    
    svm_mat1 = data[:,fing1_ind].T
    svm_mat2 = data[:,fing2_ind].T
    svm_mat = np.vstack((svm_mat1,svm_mat2))
    
    actflow_mat1 = actflow[:,fing1_ind].T
    actflow_mat2 = actflow[:,fing2_ind].T
    actflow_mat = np.vstack((actflow_mat1, actflow_mat2))

    # Spatially demean matrix across features
    samplemean = np.mean(svm_mat,axis=1)
    samplemean.shape = (len(samplemean),1)
    svm_mat = svm_mat - samplemean
    
    actflowmean = np.mean(actflow_mat,axis=1)
    actflowmean.shape = (len(actflow_mat),1)
    actflow_mat = actflow_mat - actflowmean

    scores = randomSplitLOOBaselineCV(ncvs, svm_mat, actflow_mat, labels)

    return scores

def randomSplitLOOBaselineCV(ncvs, svm_mat, actflow_mat, labels):
    """
    Runs cross validation for a within-subject SVM analysis
    Using boot-strapped CV
    Approx. 80% train set, 20% test set
    """
    
    # Data set might be unbalanced, so find minimium number of unique samples
    maxpossible = len(labels)
    for i in np.unique(labels):
        if np.sum(labels==i)<maxpossible:
            maxpossible = np.sum(labels==i)
    min_unique_samples = maxpossible
    # Train set is approximately 80%
    n_trainset_per_cond = np.floor(min_unique_samples*.8)
    # Test set is the remaining samples
    n_testset_per_cond = min_unique_samples - n_trainset_per_cond
    
    accuracies = []
    for cv in range(ncvs):
        # Define training and test set labels
        train_ind = []
        trainlabels = []
        for i in np.unique(labels):
            ind = np.where(labels==i)[0]
            train_ind.extend(np.random.choice(ind,int(n_trainset_per_cond),replace=False))
            trainlabels.extend(np.repeat(i,n_trainset_per_cond))
        train_ind = np.asarray(train_ind)
        test_ind = np.delete(np.arange(len(labels)),train_ind)
        testlabels = np.delete(labels,train_ind)
        
        # Define train set and test set matrices
        trainset = svm_mat[train_ind,:]
        testset = actflow_mat[test_ind,:]
        
        # Normalize trainset and testset using trainset stats
        mean = np.mean(svm_mat[train_ind,:],axis=0)
        mean.shape = (1,len(mean))
        std = np.std(svm_mat[train_ind,:],axis=0)
        std.shape = (1,len(std))

        trainset = np.divide((trainset - mean),std)
        
        # normalize test set using training labels from actflow data set
        mean = np.mean(actflow_mat[train_ind,:],axis=0)
        mean.shape = (1,len(mean))
        std = np.std(actflow_mat[train_ind,:],axis=0)
        std.shape = (1,len(std))
        
        testset = np.divide((testset - mean),std)

         ## Feature selection and downsampling
        unique_labels = np.unique(labels)
        feat1_labs = np.where(trainlabels==unique_labels[0])[0]
        feat2_labs = np.where(trainlabels==unique_labels[1])[0]
        # Perform t-test
        t, p = stats.ttest_rel(trainset[feat1_labs,:],trainset[feat2_labs,:],axis=0)
        h0, qs = mc.fdrcorrection0(p)
 #         h0 = p<0.1
 #         # Construct feature masks
 #         feat1_mask = np.multiply(t>0,h0).astype(bool)
 #         feat2_mask = np.multiply(t<0,h0).astype(bool)
        feat1_mask = t>0
        feat2_mask = t<0
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
            accuracies.append(_decoding((trainset,testset,trainlabels,testlabels)))
        else:
            accuracies.append(_decoding((trainset_downsampled,testset_downsampled,trainlabels,testlabels)))
        
#        accuracies.append(_decoding((trainset,testset,trainlabels,testlabels)))
        
    return np.mean(accuracies)

def _decoding((trainset,testset,trainlabels,testlabels)):
# #     clf = sklearn.linear_model.LogisticRegression()
#     clf = svm.SVC(C=1.0, kernel='linear')

#     clf.fit(trainset,trainlabels)
#     predictions = clf.predict(testset)
#     acc = predictions==testlabels
#     acc = np.mean(acc)
    unique_cond = np.unique(trainlabels)
    rdm = np.zeros((len(unique_cond),len(unique_cond)))
    acc = []
    for cond1 in unique_cond:
        mismatches = []
        prototype_ind = np.where(trainlabels==cond1)[0]
#         prototype_ind = np.random.choice(prototype_ind,size=200,replace=True)
        prototype = np.mean(trainset[prototype_ind,:],axis=0)
        for cond2 in unique_cond:
            test_ind = np.where(testlabels==cond2)[0]
#             test_ind = np.random.choice(test_ind,size=100,replace=True)
            test = np.mean(testset[test_ind,:],axis=0)
            if cond1 == cond2: 
                correct = stats.pearsonr(prototype,test)[0]
#                 correct = stats.spearmanr(prototype,test)[0]
            else:
                mismatches.append(stats.pearsonr(prototype,test)[0])
#                 mismatches.append(stats.spearmanr(prototype,test)[0])
        
        if correct > np.max(mismatches): 
            acc.append(1.0)
        else:
            acc.append(0.0)
    
    return acc


