import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import scipy.stats as stats
import os
os.environ['OMP_NUM_THREADS'] = str(1)
import statsmodels.sandbox.stats.multicomp as mc
import seaborn as sns
import h5py
import tools_group_rsa as tools_group
import nibabel as nib
import EmpiricalSRActFlow_ANN_RSA_v2 as esr
sns.set_style("whitegrid")
plt.rcParams["font.family"] = "FreeSans"
import pandas as pd
import time


## Set basic parameters
subjNums = ['013','014','016','017','018','021','023','024','026','027','028','030','031','032','033',
            '034','035','037','038','039','040','041','042','043','045','046','047','048','049','050',
            '053','055','056','057','058','062','063','066','067','068','069','070','072','074','075',
            '076','077','081','085','086','087','088','090','092','093','094','095','097','098','099',
            '101','102','103','104','105','106','108','109','110','111','112','114','115','117','119',
            '120','121','122','123','124','125','126','127','128','129','130','131','132','134','135',
            '136','137','138','139','140','141']

basedir = '/projects3/SRActFlow/'

# Using final partition
networkdef = np.loadtxt('/projects3/NetworkDiversity/data/network_partition.txt')
# network mappings for final partition set
networkmappings = {'fpn':7, 'vis1':1, 'vis2':2, 'smn':3, 'aud':8, 'lan':6, 'dan':5, 'con':4, 'dmn':9, 
                           'pmulti':10, 'none1':11, 'none2':12}

## General parameters/variables
nParcels = 360
nSubjs = len(subjNums)

glasserfile2 = '/projects/AnalysisTools/ParcelsGlasser2016/Q1-Q6_RelatedParcellation210.LR.CorticalAreas_dil_Colors.32k_fs_RL.dlabel.nii'
glasser2 = nib.load(glasserfile2).get_data()
glasser2 = np.squeeze(glasser2)


####
# Identify target vertices
# Set indices for layer-by-layer vertices
targetdir = '/projects3/SRActFlow/data/results/GroupfMRI/MotorResponseDecoding/'
motor_resp_regions_LH = np.loadtxt(targetdir + 'MotorResponseRegions_LH.csv',delimiter=',')
motor_resp_regions_RH = np.loadtxt(targetdir + 'MotorResponseRegions_RH.csv',delimiter=',')
targetROIs = np.hstack((motor_resp_regions_LH,motor_resp_regions_RH))

target_ind = []
for roi in targetROIs:
    roi_ind = np.where(glasser2==roi+1)[0]
    target_ind.extend(roi_ind)
target_ind = np.asarray(target_ind)


####
# Load in Motor Response Data
nResponses = 2
data_task_rh = np.zeros((len(glasser2),nResponses,len(subjNums)))
data_task_lh = np.zeros((len(glasser2),nResponses,len(subjNums)))

scount = 0
for subj in subjNums:
    if scount%20==0: print 'Loading motor response data for subject', scount, '/', len(subjNums)
    data_task_rh[:,:,scount] = np.real(tools_group.loadMotorResponses(subj,hand='Right'))
    data_task_lh[:,:,scount] = np.real(tools_group.loadMotorResponses(subj,hand='Left'))
    scount += 1





#####
# Load in trial data
esr = reload(esr)
#filename='/projects3/SRActFlow/data/results/GroupfMRI/RSA/EmpiricalSRActFlow_AllTrialKeys_15stims_v2.csv' # Good
filename='/projects3/SRActFlow/data/results/GroupfMRI/RSA/EmpiricalSRActFlow_AllTrialKeys_15stims_v3.csv' # Great
# filename='/projects3/SRActFlow/data/results/GroupfMRI/RSA/tmp.csv'
# esr.constructTasks(n_stims=15,filename=filename)

trial_metadata = pd.read_csv(filename)


#### Load in wrapper function
esr = reload(esr)
def subjSRActFlow_PCFC((subj,trial_metadata)):
    obj = esr.EmpiricalActFlow(subj)
    # Input
    obj.fc_input2hidden = fc_input2hidden
    obj.eig_input2hidden = eig_input2hidden
    # Rules
    obj.fc_logic2hidden = fc_logic2hidden
    obj.eig_logic2hidden = eig_logic2hidden
    obj.fc_sensory2hidden = fc_sensory2hidden
    obj.eig_sensory2hidden = eig_sensory2hidden
    obj.fc_motor2hidden = fc_motor2hidden
    obj.eig_motor2hidden = eig_motor2hidden
    # hidden 2 motor
    obj.fc_hidden2motorresp = fc_hidden2motorresp
    obj.eig_hidden2motorresp = eig_hidden2motorresp

    obj.extractAllActivations(trial_metadata)

    actflow = obj.generateActFlowPredictions_PCFC(thresh=0,verbose=False)
    del obj
    return actflow




#################################
## Beginning real permutation test
npermutations = 1000
nproc = 8
rewire='all'

print('Beginning real test...')
####
# Load FC Mappings
print('Loading FC...')

global fc_input2hidden
global fc_logic2hidden
global fc_sensory2hidden
global fc_motor2hidden
global fc_hidden2motorresp

tools_group = reload(tools_group)
inputtypes = ['color','ori','pitch','constant']
inputkeys = ['RED','VERTICAL','HIGH','CONSTANT']
fc_input2hidden = {}
eig_input2hidden = {}
i = 0
for inputtype in inputtypes:
    fc_input2hidden[inputkeys[i]], eig_input2hidden[inputkeys[i]] = tools_group.loadGroupActFlowFC(inputtype,pc_space=True)
    i += 1

# Load rules to hidden FC mappings
fc_logic2hidden, eig_logic2hidden = tools_group.loadGroupActFlowFC('Logic',pc_space=True)
fc_sensory2hidden, eig_sensory2hidden = tools_group.loadGroupActFlowFC('Sensory',pc_space=True)
fc_motor2hidden, eig_motor2hidden = tools_group.loadGroupActFlowFC('Motor',pc_space=True)

# Load hidden to motor resp mappings
fc_hidden2motorresp, eig_hidden2motorresp = tools_group.loadGroupActFlowFC('hidden2out',pc_space=True)


accuracy_rh = []
accuracy_lh = []
for i in range(npermutations):
    print 'Running permutation', i,'/',npermutations
    ## Shuffle connectivity
    conn_ind = np.arange(fc_logic2hidden.shape[1]) # Identify number of hidden layer vertices
    np.random.shuffle(conn_ind)
    if rewire=='stim':
        for stim in fc_input2hidden:
            input_ind = np.arange(fc_input2hidden[stim].shape[0]) 
            np.random.shuffle(input_ind)
            fc_input2hidden[stim] = fc_input2hidden[stim][input_ind,conn_ind]
    elif rewire=='logic':
        input_ind = np.arange(fc_logic2hidden.shape[0]) 
        np.random.shuffle(input_ind)
        fc_logic2hidden = fc_logic2hidden[input_ind,conn_ind]
    elif rewire=='sensory':
        input_ind = np.arange(fc_sensory2hidden.shape[0]) 
        np.random.shuffle(input_ind)
        fc_sensory2hidden = fc_sensory2hidden[input_ind,conn_ind]
    elif rewire=='motor':
        input_ind = np.arange(fc_motor2hidden.shape[0]) 
        np.random.shuffle(input_ind)
        fc_motor2hidden = fc_motor2hidden[input_ind,conn_ind]
    elif rewire=='all':
        for stim in fc_input2hidden:
            # Shuffle connections for stim inputs
            conn_ind = np.arange(fc_logic2hidden.shape[1]) # Identify number of hidden layer vertices
            np.random.shuffle(conn_ind)
            input_ind = np.arange(fc_input2hidden[stim].shape[0]) 
            np.random.shuffle(input_ind)
            conn_ind.shape = (1,len(conn_ind))
            input_ind.shape = (len(input_ind),1)
            fc_input2hidden[stim] = np.squeeze(fc_input2hidden[stim][:,conn_ind])
        # Shuffle connections for logic rules
        conn_ind = np.arange(fc_logic2hidden.shape[1]) # Identify number of hidden layer vertices
        np.random.shuffle(conn_ind)
        input_ind = np.arange(fc_logic2hidden.shape[0]) 
        np.random.shuffle(input_ind)
        conn_ind.shape = (1,len(conn_ind))
        input_ind.shape = (len(input_ind),1)
        fc_logic2hidden = np.squeeze(fc_logic2hidden[:,conn_ind])
        # Shuffle connections for motor rules
        conn_ind = np.arange(fc_logic2hidden.shape[1]) # Identify number of hidden layer vertices
        np.random.shuffle(conn_ind)
        input_ind = np.arange(fc_motor2hidden.shape[0]) 
        np.random.shuffle(input_ind)
        conn_ind.shape = (1,len(conn_ind))
        input_ind.shape = (len(input_ind),1)
        fc_motor2hidden = np.squeeze(fc_motor2hidden[:,conn_ind])
        # Shuffle connections for sensory rules
        conn_ind = np.arange(fc_logic2hidden.shape[1]) # Identify number of hidden layer vertices
        np.random.shuffle(conn_ind)
        input_ind = np.arange(fc_sensory2hidden.shape[0]) 
        np.random.shuffle(input_ind)
        conn_ind.shape = (1,len(conn_ind))
        input_ind.shape = (len(input_ind),1)
        fc_sensory2hidden = np.squeeze(fc_sensory2hidden[:,conn_ind])

        # Shuffle connections from hidden to motor response units
        conn_ind = np.arange(fc_hidden2motorresp.shape[1]) # Identify number of hidden layer vertices
        np.random.shuffle(conn_ind)
        input_ind = np.arange(fc_hidden2motorresp.shape[0]) 
        np.random.shuffle(input_ind)
        conn_ind.shape = (1,len(conn_ind))
        input_ind.shape = (len(input_ind),1)
        fc_hidden2motorresp = np.squeeze(fc_hidden2motorresp[:,conn_ind])

    inputs = []
    for i in range(len(subjNums)):
        inputs.append((subjNums[i],trial_metadata))

    timestart = time.time()
    pool = mp.Pool(processes=nproc)
    results = pool.map_async(subjSRActFlow_PCFC,inputs).get()
    pool.close()
    pool.join()
    timeend = time.time()

    actflow_predictions = np.zeros((len(subjNums),len(target_ind),4))
    scount = 0
    for result in results:
        actflow_predictions[scount,:,:] = result
        scount += 1


    ####
    # Sort actflow predictions into arrays
    scount = 0
    actflow_rh = np.zeros(data_task_rh.shape)
    actflow_lh = np.zeros(data_task_lh.shape)
    for scount in range(len(subjNums)):
        # RMID
        actflow_rh[target_ind,0,scount] = actflow_predictions[scount,:,2]
        # RIND
        actflow_rh[target_ind,1,scount] = actflow_predictions[scount,:,3]
        # LMID
        actflow_lh[target_ind,0,scount] = actflow_predictions[scount,:,0]
        # LIND
        actflow_lh[target_ind,1,scount] = actflow_predictions[scount,:,1]

    #### Load in target mask for RH
    tmp = np.squeeze(nib.load('/projects3/SRActFlow/data/results/GroupfMRI/MotorRegionsMasksPerSubj/sractflow_smn_outputRH_mask.dscalar.nii').get_data())
    targetmask_ind = np.where(tmp==True)[0]


    #####
    # Run decoding
    tools_group = reload(tools_group)
    nproc = 20
    nResponses = 2
    ncvs = 1

    # rois = np.asarray([8,52,9])-1
    # rois = np.asarray([8,52])-1
    rois = np.where(networkdef==networkmappings['smn'])[0]
    roi_ind = []
    for roi in rois:
        roi_ind.extend(np.where(glasser2==roi+1)[0])

    roi_ind = np.intersect1d(targetmask_ind,roi_ind)
    # roi_ind = targetmask_ind

    # realdata = stats.zscore(data_task_rh[target_ind,:,:],axis=0).copy()
    # flowdata = stats.zscore(actflow_rh[target_ind,:,:],axis=0).copy()
    realdata = data_task_rh[roi_ind,:,:].copy()
    flowdata = actflow_rh[roi_ind,:,:].copy()


    distances_baseline_rh = np.zeros((1,len(subjNums)*nResponses))
    distances_baseline_rh[0,:],rmatch,rmismatch = tools_group.actflowDecodings(realdata,
                                                                               flowdata,featsel=False,effects=True,
                                                                               ncvs=ncvs, nproc=nproc)
    accuracy_rh.append(np.mean(distances_baseline_rh[0,:]))


    ####
    # Load in target mask for LH
    tmp = np.squeeze(nib.load('/projects3/SRActFlow/data/results/GroupfMRI/MotorRegionsMasksPerSubj/sractflow_smn_outputLH_mask.dscalar.nii').get_data())

    targetmask_ind = np.where(tmp==True)[0]

    nproc = 20
    ncvs = 1

    # rois = np.asarray([188,189,232]) - 1
    # rois = np.asarray([188,232]) - 1
    # rois = np.asarray([189]) - 1
    rois = np.where(networkdef==networkmappings['smn'])[0]
    roi_ind = []
    for roi in rois:
        roi_ind.extend(np.where(glasser2==roi+1)[0])

    roi_ind = np.intersect1d(targetmask_ind,roi_ind)
    # roi_ind = targetmask_ind

    # realdata = stats.zscore(data_task_lh[roi_ind,:,:],axis=0).copy()
    # flowdata = stats.zscore(actflow_lh[roi_ind,:,:],axis=0).copy()
    realdata = data_task_lh[roi_ind,:,:].copy()
    flowdata = actflow_lh[roi_ind,:,:].copy()


    distances_baseline_lh = np.zeros((1,len(subjNums)*nResponses))
    distances_baseline_lh[0,:],rmatch,rmismatch = tools_group.actflowDecodings(realdata,
                                                                               flowdata,featsel=False,effects=True,
                                                                               ncvs=ncvs, nproc=nproc)

    accuracy_lh.append(np.mean(distances_baseline_lh[0,:]))
        
    print 'RH accuracy', np.mean(distances_baseline_rh[0,:])
    print 'LH accuracy', np.mean(distances_baseline_lh[0,:])

np.savetxt('PermutationTest_ConnectivityRewire_RH_' + rewire + '.csv', np.asarray(accuracy_rh), delimiter=',')
np.savetxt('PermutationTest_ConnectivityRewire_LH_' + rewire + '.csv', np.asarray(accuracy_lh), delimiter=',')


