# Taku Ito
# 3/18/2019
# General object to run empirical sr actflow process
# For group-level/cross-subject analyses

import numpy as np
import os
import multiprocessing as mp
import scipy.stats as stats
import nibabel as nib
import os
os.environ['OMP_NUM_THREADS'] = str(1)
import sklearn
from scipy import signal
import h5py
import sys
sys.path.append('glmScripts/')
import glmScripts.taskGLMPipeline_v2 as tgp
import sys
import pandas as pd
import pathlib
import calculateFC as fc
import tools

# Using final partition
networkdef = np.loadtxt('/home/ti61/f_mc1689_1/NetworkDiversity/data/network_partition.txt')
networkorder = np.asarray(sorted(range(len(networkdef)), key=lambda k: networkdef[k]))
networkorder.shape = (len(networkorder),1)
# network mappings for final partition set
networkmappings = {'fpn':7, 'vis1':1, 'vis2':2, 'smn':3, 'aud':8, 'lan':6, 'dan':5, 'con':4, 'dmn':9, 
                           'pmulti':10, 'none1':11, 'none2':12}
networks = networkmappings.keys()

## General parameters/variables
nParcels = 360



class Model():
    """
    Class to perform empirical actflow for a given subject (stimulus-to-response)
    """
    def __init__(self,projectdir='/home/ti61/f_mc1689_1/SRActFlow/',ruletype='12',n_hiddenregions=10,randomize=False,scratchfcdir=None):
        """
        instantiate:
            indices for condition types
            indices for specific condition instances
            betas
        """
        #### Set up basic model parameters
        self.projectdir = projectdir
        # Excluding 084
        self.subjNums = ['013','014','016','017','018','021','023','024','026','027','028','030','031','032','033',
                         '034','035','037','038','039','040','041','042','043','045','046','047','048','049','050',
                         '053','055','056','057','058','062','063','066','067','068','069','070','072','074','075',
                         '076','077','081','085','086','087','088','090','092','093','094','095','097','098','099',
                         '101','102','103','104','105','106','108','109','110','111','112','114','115','117','119',
                         '120','121','122','123','124','125','126','127','128','129','130','131','132','134','135',
                         '136','137','138','139','140','141']

        self.inputtypes = ['RED','VERTICAL','CONSTANT','HIGH']
        self.ruletype = ruletype
        #### Load in atlas
        glasserfile2 = projectdir + 'data/Q1-Q6_RelatedParcellation210.LR.CorticalAreas_dil_Colors.32k_fs_RL.dlabel.nii'
        glasser2 = nib.load(glasserfile2).get_data()
        glasser2 = np.squeeze(glasser2)
        self.glasser2 = glasser2


        #### 
        # Define hidden units
        if n_hiddenregions!=None:
            #######################################
            #### Select hidden layer regions
            hiddendir = projectdir + 'data/results/MAIN/RSA/'
            hiddenregions = np.loadtxt(hiddendir + 'RSA_Similarity_SortedRegions2.txt',delimiter=',')

            #######################################
            #### Output directory
            if randomize:
                print("Constructing model with", n_hiddenregions, "randomly selected hidden regions")
                fcdir = scratchfcdir 
                #### Necessary to optimize amarel
                pathlib.Path(fcdir).mkdir(parents=True, exist_ok=True) # Make sure directory exists
                hiddenregions = np.random.choice(hiddenregions,size=n_hiddenregions,replace=False)
            else:
                print("Constructing model with", n_hiddenregions, "hidden regions")
                fcdir = projectdir + 'data/results/MAIN/fc/LayerToLayerFC_' + str(n_hiddenregions) + 'Hidden/'
                pathlib.Path(fcdir).mkdir(parents=True, exist_ok=True) # Make sure directory exists
                # Select hidden layer
                if n_hiddenregions < 0:
                    hiddenregions = hiddenregions[n_hiddenregions:]
                else:
                    hiddenregions = hiddenregions[:n_hiddenregions]

            ## Set object attributes
            self.n_hiddenregions = n_hiddenregions
            self.hiddenregions = np.squeeze(hiddenregions)
            self.fcdir = fcdir
            self.hidden = True # Set this variable to true - indicates to run sr simulations with a hidden layer

            #### identify hidden region vertex indices
            hidden_ind = []
            for roi in hiddenregions:
                hidden_ind.extend(np.where(self.glasser2==roi+1)[0])
            self.hidden_ind = hidden_ind
        else:
            print("Constructing model with NO hidden layers")
            fcdir = projectdir + 'data/results/MAIN/fc/LayerToLayerFC_NoHidden/'
            pathlib.Path(fcdir).mkdir(parents=True, exist_ok=True) # Make sure directory exists
            self.hidden = False # Set this variable to true - indicates to run sr simulations with a hidden layer
            self.fcdir = fcdir
            self.hiddenregions = None
            self.n_hiddenregions = n_hiddenregions

        ####
        # Define task rule (input) layer
        ruledir = self.projectdir + 'data/results/MAIN/RuleDecoding/'
        if ruletype=='12':
            rule_regions = np.loadtxt(ruledir + self.ruletype + 'Rule_Regions.csv',delimiter=',')
        elif ruletype=='fpn':
            rule_regions = []
            rule_regions.extend(np.where(networkdef==networkmappings['fpn'])[0])
            rule_regions = np.asarray(rule_regions)
        elif ruletype=='nounimodal':
            allrule_regions = np.loadtxt(ruledir + '12Rule_Regions.csv',delimiter=',')
            unimodal_nets = ['vis1','aud']
            unimodal_regions = []
            for net in unimodal_nets:
                unimodal_regions.extend(np.where(networkdef==networkmappings[net])[0])
            # only include regions that are in allrule_regions but also NOT in unimodal_regions
            rule_regions = []
            for roi in allrule_regions:
                if roi in unimodal_regions:
                    continue
                else:
                    rule_regions.append(roi)
            rule_regions = np.asarray(rule_regions)


        rule_ind = []
        for roi in rule_regions:
            rule_ind.extend(np.where(self.glasser2==roi+1)[0])
        self.rule_ind = rule_ind

        #### 
        # Define motor regions
        # Set indices for layer-by-layer vertices
        targetdir = projectdir + 'data/results/MAIN/MotorResponseDecoding/'
        motor_resp_regions_LH = np.loadtxt(targetdir + 'MotorResponseRegions_LH.csv',delimiter=',')
        motor_resp_regions_RH = np.loadtxt(targetdir + 'MotorResponseRegions_RH.csv',delimiter=',')
        targetROIs = np.hstack((motor_resp_regions_LH,motor_resp_regions_RH))

        # Define all motor_ind
        motor_ind = []
        for roi in targetROIs:
            roi_ind = np.where(glasser2==roi+1)[0]
            motor_ind.extend(roi_ind)

        motor_ind = np.asarray(motor_ind).copy()
        self.motor_ind = motor_ind

        #### override -- only pick the motor parcel with the greatest response decoding
        
        motor_ind_lh = []
        for roi in motor_resp_regions_LH:
            # only include left hand responses in the right hemisphere
            if roi>=180:
                roi_ind = np.where(glasser2==roi+1)[0]
                motor_ind_lh.extend(roi_ind)

        motor_ind_rh = []
        for roi in motor_resp_regions_RH:
            # only include left hand responses in the right hemisphere
            if roi<180:
                roi_ind = np.where(glasser2==roi+1)[0]
                motor_ind_rh.extend(roi_ind)

        # 
        motor_ind_rh = np.asarray(motor_ind_rh).copy()
        motor_ind_lh = np.asarray(motor_ind_lh).copy()
        self.motor_ind_rh = motor_ind_rh
        self.motor_ind_lh = motor_ind_lh

        #### Load model task set
        filename= projectdir + 'data/results/MAIN/EmpiricalSRActFlow_AllTrialKeys_15stims_v3.csv' # Great
        self.trial_metadata = pd.read_csv(filename)

    def computeGroupFC(self,n_components=500,nproc='max'):
        """
        Function that wraps _computeSubjFC() to compute FC for all subjs, and computes averaged groupFC
        """ 
        if nproc=='max':
            nproc=mp.cpu_count()

        inputs = []
        for subj in self.subjNums:
            inputs.append((subj,n_components))

        pool = mp.Pool(processes=nproc)
        if self.hidden:
            pool.starmap_async(self._computeSubjFC,inputs)
        else:
            pool.starmap_async(self._computeSubjFC_NoHidden,inputs)
        pool.close()
        pool.join()

        #### Compute group FC
        for inputtype in self.inputtypes:
            if self.hidden:
                fc.computeGroupFC(inputtype,self.fcdir)
            else:
                fc.computeGroupFC_NoHidden(inputtype,self.fcdir)
        if self.hidden:
            fc.computeGroupFC(self.ruletype,self.fcdir)
        else:
            fc.computeGroupFC_NoHidden(self.ruletype,self.fcdir)

    def loadRealMotorResponseActivations(self,vertexmasks=True):
        #### Load motor response activations localized in output vertices only (for faster loading)
        if vertexmasks:
            print('Load real motor responses in output vertices')
            self.data_task_rh, self.data_task_lh = tools.loadMotorResponsesOutputMask()
        else:
            print('Load real motor responses in output parcels -- inefficient since need to load all vertices first')
            data_task_rh = []
            data_task_lh = []
            for subj in self.subjNums:
                tmp_rh = tools.loadMotorResponses(subj,hand='Right')
                tmp_lh = tools.loadMotorResponses(subj,hand='Left')
                data_task_rh.append(tmp_rh[self.motor_ind_rh,:].copy().T)
                data_task_lh.append(tmp_lh[self.motor_ind_lh,:].copy().T)
            self.data_task_rh = np.asarray(data_task_rh).T
            self.data_task_lh = np.asarray(data_task_lh).T

    def loadModelFC(self):
        if self.hidden:
            print('Load Model FC weights')
            fcdir = self.fcdir

            self.fc_input2hidden = {}
            self.eig_input2hidden = {}
            for inputtype in ['VERTICAL','RED','HIGH','CONSTANT']:
                self.fc_input2hidden[inputtype], self.eig_input2hidden[inputtype] = tools.loadGroupActFlowFC(inputtype,fcdir)

            # Load rule to hidden
            self.fc_12rule2hidden, self.eig_12rule2hidden = tools.loadGroupActFlowFC(self.ruletype,fcdir)
            # Load hidden to motor resp mappings
            self.fc_hidden2motorresp, self.eig_hidden2motorresp = tools.loadGroupActFlowFC('hidden2out',fcdir)
        else:
            print('Load Model FC weights -- No hidden layer')
            fcdir = self.fcdir

            self.fc_input2output = {}
            self.eig_input2output = {}
            for inputtype in ['VERTICAL','RED','HIGH','CONSTANT']:
                self.fc_input2output[inputtype], self.eig_input2output[inputtype] = tools.loadGroupActFlowFC_NoHidden(inputtype,fcdir)

            # Load rule to hidden
            self.fc_12rule2output, self.eig_12rule2output = tools.loadGroupActFlowFC_NoHidden('12',fcdir)

    def simulateGroupActFlow(self,thresh=0,nproc='max',vertexmasks=True):
        """
        Simulate group level actflow (all subject simulations)
        """
        
        if nproc=='max':
            nproc=mp.cpu_count()

        inputs = []
        for subj in self.subjNums:
            inputs.append((subj,thresh))
        
        if nproc == 1:
            results = []
            for input1 in inputs:
                results.append(self._simulateSubjActFlow(input1[0],input1[1]))
        else:
            pool = mp.Pool(processes=nproc)
            results = pool.starmap_async(self._simulateSubjActFlow,inputs).get()
            pool.close()
            pool.join()

        actflow_predictions = np.zeros((len(self.subjNums),len(self.motor_ind),4))
        #actflow_predictions_noReLU = np.zeros((len(self.subjNums),len(self.motor_ind),4))
        scount = 0
        for result in results:
        #    actflow_predictions[scount,:,:] = result[0]
        #    actflow_predictions_noReLU[scount,:,:] = result[1]
            actflow_predictions[scount,:,:] = result
            scount += 1

        ## Reformat to fit shape of actual data array
        actflow_rh = np.zeros((len(self.glasser2),2,len(self.subjNums)))
        actflow_lh = np.zeros((len(self.glasser2),2,len(self.subjNums)))
        for scount in range(len(self.subjNums)):
            # RMID
            actflow_rh[self.motor_ind,0,scount] = actflow_predictions[scount,:,2]
            # RIND
            actflow_rh[self.motor_ind,1,scount] = actflow_predictions[scount,:,3]
            # LMID
            actflow_lh[self.motor_ind,0,scount] = actflow_predictions[scount,:,0]
            # LIND
            actflow_lh[self.motor_ind,1,scount] = actflow_predictions[scount,:,1]

        #### Now save out only relevant output mask vertices
        if vertexmasks:
            tmp = np.squeeze(nib.load(self.projectdir + 'data/results/MAIN/MotorRegionsMasksPerSubj/sractflow_smn_outputRH_mask.dscalar.nii').get_data())
            rh_ind = np.where(tmp==True)[0]
            actflow_rh = actflow_rh[rh_ind,:,:]

            tmp = np.squeeze(nib.load(self.projectdir + 'data/results/MAIN/MotorRegionsMasksPerSubj/sractflow_smn_outputLH_mask.dscalar.nii').get_data())
            lh_ind = np.where(tmp==True)[0]
            actflow_lh = actflow_lh[lh_ind,:,:].copy()
        else:
            actflow_rh = actflow_rh[self.motor_ind_rh,:,:].copy()
            actflow_lh = actflow_lh[self.motor_ind_lh,:,:].copy()

        return actflow_rh, actflow_lh

    def actflowDecoding(self,trainset,testset,outputfile,
                        nbootstraps=1000,featsel=False,nproc='max',null=False,verbose=True):
        if nproc=='max':
            nproc=mp.cpu_count()

        # Decoding
        for i in range(nbootstraps):
            distances_baseline = np.zeros((1,len(self.subjNums)*2)) # subjs * nlabels
            distances_baseline[0,:],rmatch,rmismatch, confusion_mats = tools.actflowDecodings(testset,trainset,
                                                                                              effects=True, featsel=featsel,confusion=True,permutation=null,
                                                                                              ncvs=1, nproc=nproc)

            ##### Save out and append file
            # Open/create file
            filetxt = open(outputfile,"a+")
            # Write out to file
            print(np.mean(distances_baseline),file=filetxt)
            # Close file
            filetxt.close()
            
            if i%100==0 and verbose==True:
                print('Permutation', i)
                print('\tDecoding accuracy:', np.mean(distances_baseline), '| R-match:', np.mean(rmatch), '| R-mismatch:', np.mean(rmismatch))

    def extractSubjActivations(self, subj, df_trials):
        """
        extract activations for a sample subject, including motor response
        """

        ## Set up data parameters
        X = tgp.loadTaskTiming(subj,'ALL')
        self.stimIndex = np.asarray(X['stimIndex'])
        self.stimCond = np.asarray(X['stimCond'])

        datadir = self.projectdir + 'data/postProcessing/hcpPostProcCiric/'
        h5f = h5py.File(datadir + subj + '_glmOutput_data.h5','r')
        self.betas = h5f['taskRegression/ALL_24pXaCompCorXVolterra_taskReg_betas_canonical'][:].copy()
        h5f.close()

        ## Set up task parameters
        self.logicRules = ['BOTH', 'NOTBOTH', 'EITHER', 'NEITHER']
        self.sensoryRules = ['RED', 'VERTICAL', 'HIGH', 'CONSTANT']
        self.motorRules = ['LMID', 'LIND', 'RMID', 'RIND']
        self.colorStim = ['RED', 'BLUE']
        self.oriStim = ['VERTICAL', 'HORIZONTAL']
        self.pitchStim = ['HIGH', 'LOW']
        self.constantStim = ['CONSTANT','ALARM']


        # Begin extraction for specific trials
        n_trials = len(df_trials)
        
        stimData = np.zeros((n_trials,self.betas.shape[0]))
        logicRuleData = np.zeros((n_trials,self.betas.shape[0]))
        sensoryRuleData = np.zeros((n_trials,self.betas.shape[0]))
        motorRuleData = np.zeros((n_trials,self.betas.shape[0]))
        respData = np.zeros((n_trials,self.betas.shape[0]))
        sensoryRuleIndices = []
        motorRespAll = []
        
        for trial in range(n_trials):
            logicRule = df_trials.iloc[trial].logicRule
            sensoryRule = df_trials.iloc[trial].sensoryRule
            motorRule = df_trials.iloc[trial].motorRule
            motorResp = df_trials.iloc[trial].motorResp
            stim1 = df_trials.iloc[trial].stim1
            stim2 = df_trials.iloc[trial].stim2

#                        if verbose:
#                            print 'Running actflow predictions for:', logicRule, sensoryRule, motorRule, 'task'

            logicKey = 'RuleLogic_' + logicRule
            sensoryKey = 'RuleSensory_' + sensoryRule
            motorKey = 'RuleMotor_' + motorRule
            stimKey = 'Stim_' + stim1 + stim2
            motorResp = solveInputs(logicRule, sensoryRule, motorRule, stim1, stim2, printTask=False)
            respKey = 'Response_' + motorResp


            stimKey_ind = np.where(self.stimCond==stimKey)[0]
            logicRule_ind = np.where(self.stimCond==logicKey)[0]
            sensoryRule_ind = np.where(self.stimCond==sensoryKey)[0]
            motorRule_ind = np.where(self.stimCond==motorKey)[0]
            respKey_ind = np.where(self.stimCond==respKey)[0]


            stimData[trial,:] = np.real(self.betas[:,stimKey_ind].copy()[:,0])
            logicRuleData[trial,:] = np.real(self.betas[:,logicRule_ind].copy()[:,0])
            sensoryRuleData[trial,:] = np.real(self.betas[:,sensoryRule_ind].copy()[:,0])
            motorRuleData[trial,:] = np.real(self.betas[:,motorRule_ind].copy()[:,0])
            respData[trial,:] = np.real(self.betas[:,respKey_ind].copy()[:,0])
                        
            motorRespAll.append(motorResp)
            sensoryRuleIndices.append(sensoryRule)

        self.motorRespAll = motorRespAll
        self.stimData = stimData
        self.logicRuleData = logicRuleData
        self.sensoryRuleData = sensoryRuleData
        self.motorRuleData = motorRuleData
        self.respData = respData
        self.sensoryRuleIndices = sensoryRuleIndices

    def extractSubjHiddenRSMActivations(self, subj):
        """
        extract activations for a sample subject, including motor response
        """

        ## Set up data parameters
        X = tgp.loadTaskTiming(subj,'ALL')
        self.stimIndex = np.asarray(X['stimIndex'])
        self.stimCond = np.asarray(X['stimCond'])

        datadir = self.projectdir + 'data/postProcessing/hcpPostProcCiric/'
        h5f = h5py.File(datadir + subj + '_glmOutput_data.h5','r')
        self.betas = h5f['taskRegression/ALL_24pXaCompCorXVolterra_taskReg_betas_canonical'][:].copy()
        h5f.close()

        ## Set up task parameters
        self.logicRules = ['BOTH', 'NOTBOTH', 'EITHER', 'NEITHER']
        self.sensoryRules = ['RED', 'VERTICAL', 'HIGH', 'CONSTANT']
        self.motorRules = ['LMID', 'LIND', 'RMID', 'RIND']
        self.colorStim = ['RED', 'BLUE']
        self.oriStim = ['VERTICAL', 'HORIZONTAL']
        self.pitchStim = ['HIGH', 'LOW']
        self.constantStim = ['CONSTANT','ALARM']

        total_conds = 28 # 12 rules + 16 stimulus pairings
        rsm_activations = np.zeros((28,self.betas.shape[0]))
        labels = []
        condcount = 0
        ## 
        # START
        for cond in self.logicRules:
            labels.append(cond)
            key = 'RuleLogic_' + cond
            ind = np.where(self.stimCond==key)[0]
            rsm_activations[condcount,:] = np.real(self.betas[:,ind].copy()[:,0])
            condcount += 1 # go to next condition

        for cond in self.sensoryRules:
            labels.append(cond)
            key = 'RuleSensory_' + cond
            ind = np.where(self.stimCond==key)[0]
            rsm_activations[condcount,:] = np.real(self.betas[:,ind].copy()[:,0])
            condcount += 1 # go to next condition

        for cond in self.motorRules:
            labels.append(cond)
            key = 'RuleMotor_' + cond
            ind = np.where(self.stimCond==key)[0]
            rsm_activations[condcount,:] = np.real(self.betas[:,ind].copy()[:,0])
            condcount += 1 # go to next condition


        # This is nested for loop since stimuli come in pairs
        for cond1 in self.colorStim:
            for cond2 in self.colorStim:
                labels.append(cond1 + cond2)
                key = 'Stim_' + cond1 + cond2
                ind = np.where(self.stimCond==key)[0]
                rsm_activations[condcount,:] = np.real(self.betas[:,ind].copy()[:,0])
                condcount += 1 # go to next condition

        for cond1 in self.oriStim:
            for cond2 in self.oriStim:
                labels.append(cond1 + cond2)
                key = 'Stim_' + cond1 + cond2
                ind = np.where(self.stimCond==key)[0]
                rsm_activations[condcount,:] = np.real(self.betas[:,ind].copy()[:,0])
                condcount += 1 # go to next condition

        for cond1 in self.pitchStim:
            for cond2 in self.pitchStim:
                labels.append(cond1 + cond2)
                key = 'Stim_' + cond1 + cond2
                ind = np.where(self.stimCond==key)[0]
                rsm_activations[condcount,:] = np.real(self.betas[:,ind].copy()[:,0])
                condcount += 1 # go to next condition

        for cond1 in self.constantStim:
            for cond2 in self.constantStim:
                labels.append(cond1 + cond2)
                key = 'Stim_' + cond1 + cond2
                ind = np.where(self.stimCond==key)[0]
                rsm_activations[condcount,:] = np.real(self.betas[:,ind].copy()[:,0])
                condcount += 1 # go to next condition

        return rsm_activations, labels

    def generateHiddenUnitRSMPredictions(self,thresh=0,n_hiddenregions=10,filename='',verbose=False):
        """
        Run all predictions for all 64 tasks
        """
        hidden_ind = self.hidden_ind
        rule_ind = self.rule_ind

        all_actflow_unthresh = []
        all_actflow_thresh = []
        all_true_activity = []
        for subj in self.subjNums:
            print('Predicting hidden layer activations for subject', subj)
            rsm_activations, labels = self.extractSubjHiddenRSMActivations(subj)
            
            tmp_actflow_unthresh = []
            tmp_actflow_thresh = []
            tmp_true_activity = []
            labelcount = 0
            for label in labels:
                # Dissociate sensory rules from sensory stimuli since stimuli have two stimulus words (e.g., 'REDRED')
                if label in ['BOTH', 'NOTBOTH', 'EITHER', 'NEITHER', 'RED', 'VERTICAL', 'HIGH', 'CONSTANT', 'LMID', 'LIND', 'RMID', 'RIND']:
                    input_units = 'rule'
                if label in ['REDRED', 'REDBLUE', 'BLUERED', 'BLUEBLUE']:
                    input_units = 'RED' # specify sensory rules for sensory activations
                if label in ['VERTICALVERTICAL', 'VERTICALHORIZONTAL', 'HORIZONTALVERTICAL', 'HORIZONTALHORIZONTAL']:
                    input_units = 'VERTICAL' # this is the sensory rule
                if label in ['HIGHHIGH', 'HIGHLOW', 'LOWHIGH', 'LOWLOW']:
                    input_units = 'HIGH'
                if label in ['CONSTANTCONSTANT', 'CONSTANTALARM', 'ALARMCONSTANT', 'ALARMALARM']:
                    input_units = 'CONSTANT'

                if input_units!='rule':
                    input_ind = self._getStimIndices(input_units) # Identify the vertices for stimulus layer of the ANN 
                    unique_input_ind = np.where(np.in1d(input_ind,hidden_ind)==False)[0]

                    fc = self.fc_input2hidden[input_units]
                    pc_act = np.matmul(rsm_activations[labelcount,:][unique_input_ind],self.eig_input2hidden[input_units].T)
                    # Unthresholded actflow
                    actflow_unthresh = np.matmul(pc_act,fc) 
                    # Thresholded actflow
                    actflow_thresh = np.multiply(actflow_unthresh,actflow_unthresh>thresh)

                if input_units=='rule':
                    unique_input_ind = np.where(np.in1d(rule_ind,hidden_ind)==False)[0]
                    fc = self.fc_12rule2hidden
                    pc_act = np.matmul(rsm_activations[labelcount,:][unique_input_ind],self.eig_12rule2hidden.T)
                    # Unthresholded actflow
                    actflow_unthresh = np.matmul(pc_act,fc) 
                    # Thresholded actflow
                    actflow_thresh = np.multiply(actflow_unthresh,actflow_unthresh>thresh)

                tmp_actflow_unthresh.append(actflow_unthresh)
                tmp_actflow_thresh.append(actflow_thresh)
                tmp_true_activity.append(np.squeeze(rsm_activations[labelcount,hidden_ind]))

                labelcount += 1

            # Compute subject-specific predicted activations for each condition
            all_actflow_unthresh.append(np.asarray(tmp_actflow_unthresh))
            all_actflow_thresh.append(np.asarray(tmp_actflow_thresh))
            all_true_activity.append(np.asarray(tmp_true_activity))



        np.savetxt(filename + '.txt', labels, fmt='%s')

        h5f = h5py.File(filename + '.h5','a')
        try:
            h5f.create_dataset('actflow_unthresh',data=all_actflow_unthresh)
            h5f.create_dataset('actflow_thresh',data=all_actflow_thresh)
            h5f.create_dataset('true_activity',data=all_true_activity)
        except:
            del h5f['actflow_unthresh'], h5f['actflow_thresh'], h5f['true_activity']
            h5f.create_dataset('actflow_unthresh',data=all_actflow_unthresh)
            h5f.create_dataset('actflow_thresh',data=all_actflow_thresh)
            h5f.create_dataset('true_activity',data=all_true_activity)
        h5f.close()


    def generateInputControlDecoding(self,n_hiddenregions=10,verbose=False):
        """
        Run all predictions for all 64 tasks
        """
        hidden_ind = self.hidden_ind
        rule_ind = self.rule_ind

        
        # Also exclude smn indices
        smn_rois = np.where(networkdef==networkmappings['smn'])[0]
        smn_ind = []
        for roi in smn_rois:
            smn_ind.extend(np.where(self.glasser2==roi+1)[0])
        smn_ind = np.asarray(smn_ind)

        target_vertices = self.fc_hidden2motorresp.shape[1]
        actflow = np.zeros((target_vertices,4)) #LMID, LIND, RMID, rIND -- 4 cols in 3rd dim for each sensory rule
        input_activations_lmid = []
        input_activations_lind = []
        input_activations_rmid = []
        input_activations_rind = []
        all_input_ind = []
        for sensoryRule in self.sensoryRules:
            input_ind = self._getStimIndices(sensoryRule) # Identify the vertices for the stimulus layer of the ANN 
            all_input_ind.extend(input_ind)

        all_input_ind = np.asarray(all_input_ind)

        #### Input activations
        unique_input_ind = np.where(np.in1d(all_input_ind,hidden_ind)==False)[0]
        unique_input_ind = np.where(np.in1d(unique_input_ind,smn_ind)==False)[0]
        input_act = self.stimData[:,:][:,unique_input_ind]
        
        #### 12 rule activations
        unique_input_ind = np.where(np.in1d(rule_ind,hidden_ind)==False)[0]
        unique_input_ind = np.where(np.in1d(unique_input_ind,smn_ind)==False)[0]
        rule_composition = self.logicRuleData[:,unique_input_ind] + self.sensoryRuleData[:,unique_input_ind] + self.motorRuleData[:,unique_input_ind]
        #rule_act = self.logicRuleData[:,:][:,unique_input_ind]

        ##### Concatenate input activations
        input_activations = np.hstack((input_act,rule_composition))
        ## Apply threshold
        input_activations = np.multiply(input_act,input_act>0)


        #### Average into 4 different responses
        respIndex = np.asarray(self.motorRespAll)
        # LMID
        ind = np.where(respIndex=='LMID')[0]
        if len(ind)!=0:
            input_activations_lmid.append(np.sum(input_activations[ind,:],axis=0))
        # LIND
        ind = np.where(respIndex=='LIND')[0]
        if len(ind)!=0:
            input_activations_lind.append(np.sum(input_activations[ind,:],axis=0))
        # RMID
        ind = np.where(respIndex=='RMID')[0]
        if len(ind)!=0:
            input_activations_rmid.append(np.sum(input_activations[ind,:],axis=0))
        # RIND
        ind = np.where(respIndex=='RIND')[0]
        if len(ind)!=0:
            input_activations_rind.append(np.sum(input_activations[ind,:],axis=0))

        return input_activations_lmid, input_activations_lind, input_activations_rmid, input_activations_rind

    def generateActFlowPredictions_12Rule_PCFC(self,thresh=0,n_hiddenregions=10,verbose=False):
        """
        Run all predictions for all 64 tasks
        """
        hidden_ind = self.hidden_ind
        rule_ind = self.rule_ind

        target_vertices = self.fc_hidden2motorresp.shape[1]
        actflow = np.zeros((target_vertices,4)) #LMID, LIND, RMID, rIND -- 4 cols in 3rd dim for each sensory rule
        sensecount = 0
        for sensoryRule in self.sensoryRules:
            sensoryIndices = np.where(np.asarray(self.sensoryRuleIndices)==sensoryRule)[0]

            input_ind = self._getStimIndices(sensoryRule) # Identify the vertices for stimulus layer of the ANN 

            # Run activity flow

            #### Input to hidden regions
            # first identify non-overlapping indices
            unique_input_ind = np.where(np.in1d(input_ind,hidden_ind)==False)[0]
            fc = self.fc_input2hidden[sensoryRule]
            pc_act = np.matmul(self.stimData[sensoryIndices,:][:,unique_input_ind],self.eig_input2hidden[sensoryRule].T)
            actflow_stim = np.matmul(pc_act,fc) 
            
            #### Rule compositions
            ####  (12rule) to hidden regions
            unique_input_ind = np.where(np.in1d(rule_ind,hidden_ind)==False)[0]
            rule_composition = self.logicRuleData[sensoryIndices,:][:,unique_input_ind] + self.sensoryRuleData[sensoryIndices,:][:,unique_input_ind] + self.motorRuleData[sensoryIndices,:][:,unique_input_ind]
            # first identify non-overlapping indices
            fc = self.fc_12rule2hidden
            pc_act = np.matmul(rule_composition,self.eig_12rule2hidden.T)
            actflow_taskrules = np.matmul(pc_act,fc) 

            hiddenlayer_composition = actflow_taskrules + actflow_stim

            # Apply a threshold if there is one
            if thresh==None:
                pass
            else:
                hiddenlayer_composition = np.multiply(hiddenlayer_composition,hiddenlayer_composition>thresh)
                #t, p = stats.ttest_1samp(hiddenlayer_composition,0,axis=0) # Trials x Vertices
                #p[t>0] = p[t>0]/2.0
                #p[t<0] = 1.0 - p[t<0]/2.0
                #h0 = mc.fdrcorrection0(p)[0] # Effectively a threshold linear func
                #h0 = p<0.05
                #hiddenlayer_composition = np.multiply(hiddenlayer_composition,h0)

            ## multiplicative gating
            ##hiddenlayer_composition = np.multiply(np.multiply(np.multiply(actflow_stim, actflow_logicrule), actflow_sensoryrule), actflow_motorrule)

            #### Hidden to output regions 
            unique_ind = np.where(np.in1d(hidden_ind,self.motor_ind)==False)[0]
            fc = self.fc_hidden2motorresp
            pc_act = np.matmul(hiddenlayer_composition[:,unique_ind],self.eig_hidden2motorresp.T)
            actflow_predictions = np.real(np.matmul(pc_act,fc))


            ## Apply a threshold if there is one
            #if thresh==None:
            #    pass
            #else:
            #    actflow_predictions = np.multiply(actflow_predictions,actflow_predictions>thresh)


            respIndex = np.asarray(self.motorRespAll)[sensoryIndices]
            # LMID
            ind = np.where(respIndex=='LMID')[0]
            if len(ind)!=0:
                actflow[:,0] = actflow[:,0] + np.sum(actflow_predictions[ind,:],axis=0)
            # LIND
            ind = np.where(respIndex=='LIND')[0]
            if len(ind)!=0:
                actflow[:,1] = actflow[:,1] + np.sum(actflow_predictions[ind,:],axis=0)
            # RMID
            ind = np.where(respIndex=='RMID')[0]
            if len(ind)!=0:
                actflow[:,2] = actflow[:,2] + np.sum(actflow_predictions[ind,:],axis=0)
            # RIND
            ind = np.where(respIndex=='RIND')[0]
            if len(ind)!=0:
                actflow[:,3] = actflow[:,3] + np.sum(actflow_predictions[ind,:],axis=0)
            
            #
            #respIndex = np.asarray(self.motorRespAll)[sensoryIndices]
            ## LMID
            #ind = np.where(respIndex=='LMID')[0]
            #print ind
            #actflow[:,0,sensecount] = np.mean(actflow_predictions[ind,:],axis=0)
            ## LIND
            #ind = np.where(respIndex=='LIND')[0]
            #print ind
            #actflow[:,1,sensecount] = np.mean(actflow_predictions[ind,:],axis=0)
            ## RMID
            #ind = np.where(respIndex=='RMID')[0]
            #print ind
            #actflow[:,2,sensecount] = np.mean(actflow_predictions[ind,:],axis=0)
            ## RIND
            #ind = np.where(respIndex=='RIND')[0]
            #print ind
            #actflow[:,3,sensecount] = np.mean(actflow_predictions[ind,:],axis=0)

            sensecount += 1

        #actflow = np.mean(actflow,axis=2)
        #actflow = np.divide(actflow,4.0)

        return actflow

    def generateActFlowPredictions_12Rule_NoHidden(self,thresh=0,verbose=False):
        """
        Run all predictions for all 64 tasks
        """
        rule_ind = self.rule_ind

        target_vertices = self.fc_12rule2output.shape[1]
        actflow = np.zeros((target_vertices,4)) #LMID, LIND, RMID, rIND -- 4 cols in 3rd dim for each sensory rule
        sensecount = 0
        for sensoryRule in self.sensoryRules:
            sensoryIndices = np.where(np.asarray(self.sensoryRuleIndices)==sensoryRule)[0]

            input_ind = self._getStimIndices(sensoryRule) # Identify the vertices for stimulus layer of the ANN 

            # Run activity flow

            #### Input to hidden regions
            # first identify non-overlapping indices
            unique_input_ind = np.where(np.in1d(input_ind,self.motor_ind)==False)[0]
            fc = self.fc_input2output[sensoryRule]
            pc_act = np.matmul(self.stimData[sensoryIndices,:][:,unique_input_ind],self.eig_input2output[sensoryRule].T)
            actflow_stim = np.matmul(pc_act,fc) 
            
            #### Rule compositions
            ####  (12rule) to hidden regions
            unique_input_ind = np.where(np.in1d(rule_ind,self.motor_ind)==False)[0]
            rule_composition = self.logicRuleData[sensoryIndices,:][:,unique_input_ind] + self.sensoryRuleData[sensoryIndices,:][:,unique_input_ind] + self.motorRuleData[sensoryIndices,:][:,unique_input_ind]
            # first identify non-overlapping indices
            fc = self.fc_12rule2output
            pc_act = np.matmul(rule_composition,self.eig_12rule2output.T)
            actflow_taskrules = np.matmul(pc_act,fc) 

            actflow_predictions = actflow_taskrules + actflow_stim

#            # Apply a threshold if there is one
#            if thresh==None:
#                pass
#            else:
#                actflow_predictions = np.multiply(actflow_predictions,actflow_predictions>thresh)
                #t, p = stats.ttest_1samp(hiddenlayer_composition,0,axis=0) # Trials x Vertices
                #p[t>0] = p[t>0]/2.0
                #p[t<0] = 1.0 - p[t<0]/2.0
                #h0 = mc.fdrcorrection0(p)[0] # Effectively a threshold linear func
                #h0 = p<0.05
                #hiddenlayer_composition = np.multiply(hiddenlayer_composition,h0)

            respIndex = np.asarray(self.motorRespAll)[sensoryIndices]
            # LMID
            ind = np.where(respIndex=='LMID')[0]
            if len(ind)!=0:
                actflow[:,0] = actflow[:,0] + np.sum(actflow_predictions[ind,:],axis=0)
            # LIND
            ind = np.where(respIndex=='LIND')[0]
            if len(ind)!=0:
                actflow[:,1] = actflow[:,1] + np.sum(actflow_predictions[ind,:],axis=0)
            # RMID
            ind = np.where(respIndex=='RMID')[0]
            if len(ind)!=0:
                actflow[:,2] = actflow[:,2] + np.sum(actflow_predictions[ind,:],axis=0)
            # RIND
            ind = np.where(respIndex=='RIND')[0]
            if len(ind)!=0:
                actflow[:,3] = actflow[:,3] + np.sum(actflow_predictions[ind,:],axis=0)
            
            #
            #respIndex = np.asarray(self.motorRespAll)[sensoryIndices]
            ## LMID
            #ind = np.where(respIndex=='LMID')[0]
            #print ind
            #actflow[:,0,sensecount] = np.mean(actflow_predictions[ind,:],axis=0)
            ## LIND
            #ind = np.where(respIndex=='LIND')[0]
            #print ind
            #actflow[:,1,sensecount] = np.mean(actflow_predictions[ind,:],axis=0)
            ## RMID
            #ind = np.where(respIndex=='RMID')[0]
            #print ind
            #actflow[:,2,sensecount] = np.mean(actflow_predictions[ind,:],axis=0)
            ## RIND
            #ind = np.where(respIndex=='RIND')[0]
            #print ind
            #actflow[:,3,sensecount] = np.mean(actflow_predictions[ind,:],axis=0)

            sensecount += 1

        #actflow = np.mean(actflow,axis=2)
        #actflow = np.divide(actflow,4.0)

        return actflow

    def _simulateSubjActFlow(self,subj,thresh=0):
        print('Subject ' + subj + '... Simulating ' + str(len(self.trial_metadata)) + ' Trials')
        self.extractSubjActivations(subj,self.trial_metadata)
        if self.hidden:
            actflow = self.generateActFlowPredictions_12Rule_PCFC(thresh=thresh,verbose=False)
        else:
            actflow = self.generateActFlowPredictions_12Rule_NoHidden(thresh=thresh,verbose=False)
        return actflow

    def _getStimIndices(self,sensoryRule):
        if sensoryRule=='RED': 
            inputkey = 'COLOR'
        if sensoryRule=='VERTICAL':
            inputkey = 'ORI'
        if sensoryRule=='HIGH':
            inputkey = 'PITCH'
        if sensoryRule=='CONSTANT': 
            inputkey = 'CONSTANT'
        
        inputdir = self.projectdir + 'data/results/MAIN/InputStimuliDecoding/'
        input_regions = np.loadtxt(inputdir + 'InputStimuliRegions_' + inputkey + '.csv',delimiter=',')

        input_ind = []
        for roi in input_regions:
            input_ind.extend(np.where(self.glasser2==roi+1)[0])

        return input_ind

    def _computeSubjFC(self,subj,n_components=500):
        """
        Compute FC
        """
        # Set some useful local parameters
        fcdir = self.fcdir
        n_hiddenregions = self.n_hiddenregions
        hiddenregions = self.hiddenregions

        print('Computing FC for subject', subj, '| num hidden =', n_hiddenregions, '| num components =', n_components)

        #######################################
        #### Load data
        data = tools.loadRestActivity(subj,model='24pXaCompCorXVolterra',zscore=False)
        # Demean
        data = signal.detrend(data,axis=1,type='constant')

        #######################################
        #### Compute input 2 hidden layer FC
        print('Computing input to hidden FC | subj', subj)
        # Re-map inputtypes to sensory features
        inputmap = {'RED':'COLOR','VERTICAL':'ORI','CONSTANT':'CONSTANT','HIGH':'PITCH'}
        for inputtype in self.inputtypes:
            outputfilename = fcdir + '/' + inputtype + 'To' + 'HiddenLayer_FC_subj' + subj + '.h5'
            sourcedir = self.projectdir + 'data/results/MAIN/InputStimuliDecoding/'
            input_regions = np.loadtxt(sourcedir + 'InputStimuliRegions_' + inputmap[inputtype] + '.csv',delimiter=',')
            # Compute FC
            fc.layerToLayerFC(data,input_regions,hiddenregions,outputfilename,n_components=n_components)

        #######################################
        #### Compute rule 2 hidden layer FC
        print('Computing rule to hidden FC | subj', subj)
        outputfilename = fcdir + '/' + self.ruletype + 'RuleToHiddenLayer_FC_subj' + subj + '.h5'
        sourcedir = self.projectdir + 'data/results/MAIN/RuleDecoding/'
        if self.ruletype == 'fpn':
            rule_regions = []
            rule_regions.extend(np.where(networkdef==networkmappings['fpn'])[0])
            rule_regions = np.asarray(rule_regions)
        elif self.ruletype=='nounimodal':
            allrule_regions = np.loadtxt(sourcedir + '12Rule_Regions.csv',delimiter=',')
            unimodal_nets = ['vis1','aud']
            unimodal_regions = []
            for net in unimodal_nets:
                unimodal_regions.extend(np.where(networkdef==networkmappings[net])[0])
            # only include regions that are in allrule_regions but also NOT in unimodal_regions
            rule_regions = []
            for roi in allrule_regions:
                if roi in unimodal_regions:
                    continue
                else:
                    rule_regions.append(roi)
            rule_regions = np.asarray(rule_regions)
        else:
            rule_regions = np.loadtxt(sourcedir + self.ruletype + 'Rule_Regions.csv',delimiter=',')

        # Compute FC
        fc.layerToLayerFC(data,rule_regions,hiddenregions,outputfilename,n_components=n_components)

        #######################################
        #### Compute hidden to output layer 
        outputfilename = fcdir + '/HiddenLayerToOutput_FC_subj' + subj + '.h5'
        targetdir = self.projectdir + 'data/results/MAIN/MotorResponseDecoding/'
        motor_resp_regions_LH = np.loadtxt(targetdir + 'MotorResponseRegions_LH.csv',delimiter=',')
        motor_resp_regions_RH = np.loadtxt(targetdir + 'MotorResponseRegions_RH.csv',delimiter=',')
        motor_resp_regions = np.hstack((motor_resp_regions_LH,motor_resp_regions_RH))
        # Compute FC
        print('Computing hidden to output FC | subj', subj)
        fc.layerToLayerFC(data,hiddenregions,motor_resp_regions,outputfilename,n_components=n_components)

    def _computeSubjFC_NoHidden(self,subj,n_components=500):
        """
        Compute FC
        """
        # Set some useful local parameters
        fcdir = self.fcdir
        n_hiddenregions = self.n_hiddenregions
        hiddenregions = self.hiddenregions

        print('Computing FC for subject', subj, '| num hidden =', n_hiddenregions, '| num components =', n_components)

        #######################################
        #### Load data
        data = tools.loadRestActivity(subj,model='24pXaCompCorXVolterra',zscore=False)
        # Demean
        data = signal.detrend(data,axis=1,type='constant')

        #######################################
        #### Compute input 2 output layer FC
        print('Computing input to output (NoHidden) FC | subj', subj)
        targetdir = self.projectdir + 'data/results/MAIN/MotorResponseDecoding/'
        motor_resp_regions_LH = np.loadtxt(targetdir + 'MotorResponseRegions_LH.csv',delimiter=',')
        motor_resp_regions_RH = np.loadtxt(targetdir + 'MotorResponseRegions_RH.csv',delimiter=',')
        motor_resp_regions = np.hstack((motor_resp_regions_LH,motor_resp_regions_RH))

        # Re-map inputtypes to sensory features
        inputmap = {'RED':'COLOR','VERTICAL':'ORI','CONSTANT':'CONSTANT','HIGH':'PITCH'}
        for inputtype in self.inputtypes:
            outputfilename = fcdir + '/' + inputtype + 'To' + 'OutputLayer_FC_subj' + subj + '.h5'
            sourcedir = self.projectdir + 'data/results/MAIN/InputStimuliDecoding/'
            input_regions = np.loadtxt(sourcedir + 'InputStimuliRegions_' + inputmap[inputtype] + '.csv',delimiter=',')
            # Compute FC
            fc.layerToLayerFC(data,input_regions,motor_resp_regions,outputfilename,n_components=n_components)

        #######################################
        #### Compute rule 2 output layer FC
        print('Computing rule to output (NoHidden) FC | subj', subj)
        outputfilename = fcdir + '/' + self.ruletype + 'RuleToHiddenLayer_FC_subj' + subj + '.h5'
        sourcedir = self.projectdir + 'data/results/MAIN/RuleDecoding/'
        if self.ruletype == 'fpn':
            rule_regions = []
            rule_regions.extend(np.where(networkdef==networkmappings['fpn'])[0])
            rule_regions = np.asarray(rule_regions)
        elif self.ruletype=='nounimodal':
            allrule_regions = np.loadtxt(sourcedir + '12Rule_Regions.csv',delimiter=',')
            unimodal_nets = ['vis1','aud']
            unimodal_regions = []
            for net in unimodal_nets:
                unimodal_regions.extend(np.where(networkdef==networkmappings[net])[0])
            # only include regions that are in allrule_regions but also NOT in unimodal_regions
            rule_regions = []
            for roi in allrule_regions:
                if roi in unimodal_regions:
                    continue
                else:
                    rule_regions.append(roi)
            rule_regions = np.asarray(rule_regions)
        else:
            rule_regions = np.loadtxt(sourcedir + self.ruletype + 'Rule_Regions.csv',delimiter=',')

        # Compute FC
        fc.layerToLayerFC(data,rule_regions,motor_resp_regions,outputfilename,n_components=n_components)

def solveInputs(logicRule, sensoryRule, motorRule, stim1, stim2, printTask=False):
    """
    Solves CPRO task given a set of inputs and a task rule
    logicRule = [BOTH, NOTBOTH, EITHER, NEITHER]
    sensoryRule = [RED, VERTICAL, HIGH, CONSTANT]
    motorRule = [LMID, LIND, RMID, RIND]
    stim1 = [RED,BLUE] # for example
    stim2 = [RED,BLUE] # for example
    """

    # Run through logic rule gates
    if logicRule == 'BOTH':
        if stim1==sensoryRule and stim2==sensoryRule:
            gate = True
        else:
            gate = False

    if logicRule == 'NOTBOTH':
        if stim1!=sensoryRule or stim2!=sensoryRule:
            gate = True
        else:
            gate = False

    if logicRule == 'EITHER':
        if stim1==sensoryRule or stim2==sensoryRule:
            gate = True
        else:
            gate = False

    if logicRule == 'NEITHER':
        if stim1!=sensoryRule and stim2!=sensoryRule:
            gate = True
        else:
            gate = False


    # Apply logic gating to motor rules
    if motorRule=='LMID':
        if gate==True:
            motorOutput = 'LMID'
        else:
            motorOutput = 'LIND'

    if motorRule=='LIND':
        if gate==True:
            motorOutput = 'LIND'
        else:
            motorOutput = 'LMID'

    if motorRule=='RMID':
        if gate==True:
            motorOutput = 'RMID'
        else:
            motorOutput = 'RIND'

    if motorRule=='RIND':
        if gate==True:
            motorOutput = 'RIND'
        else:
            motorOutput = 'RMID'

    ## Print task first
    if printTask:
        print('Logic rule:', logicRule)
        print('Sensory rule:', sensoryRule)
        print('Motor rule:', motorRule)
        print('**Stimuli**')
        print(stim1, stim2)
        print('Motor response:', motorOutput)

    return motorOutput

def constructTasks(n_stims=1,filename='./EmpiricalSRActFlow_AllTrialKeys1.h5'):
    """
    Construct and save a dictionary of tasks to simulate/generate data for
   
    """
    logicRules = ['BOTH', 'NOTBOTH', 'EITHER', 'NEITHER']
    sensoryRules = ['RED', 'VERTICAL', 'HIGH', 'CONSTANT']
    motorRules = ['LMID', 'LIND', 'RMID', 'RIND']
    colorStim = ['RED', 'BLUE']
    oriStim = ['VERTICAL', 'HORIZONTAL']
    pitchStim = ['HIGH', 'LOW']
    constantStim = ['CONSTANT','ALARM']

    trial_dict = {}
    trial_dict['logicRule'] = []
    trial_dict['sensoryRule'] = []
    trial_dict['motorRule'] = []
    trial_dict['stim1'] =  []
    trial_dict['stim2'] = []
    trial_dict['motorResp'] = []
    for sensoryRule in sensoryRules:

        for nstim in range(n_stims):
            respIndex = []
            # Randomly sample two stimulus patterns depending on the sensory rule
            if sensoryRule=='RED': stims = np.random.choice(colorStim,2,replace=True)
            if sensoryRule=='VERTICAL': stims = np.random.choice(oriStim,2,replace=True)
            if sensoryRule=='HIGH': stims = np.random.choice(pitchStim,2,replace=True)
            if sensoryRule=='PITCH': stims = np.random.choice(constantStim,2,replace=True)
            stim1, stim2 = stims

            
            for logicRule in logicRules:

                for motorRule in motorRules:

#                        if verbose:
#                            print 'Running actflow predictions for:', logicRule, sensoryRule, motorRule, 'task'

                    logicKey = 'RuleLogic_' + logicRule
                    sensoryKey = 'RuleSensory_' + sensoryRule
                    motorKey = 'RuleMotor_' + motorRule
                    stimKey = 'Stim_' + stim1 + stim2
                    motorResp = solveInputs(logicRule, sensoryRule, motorRule, stim1, stim2, printTask=False)
                    respKey = 'Response_' + motorResp
                    
                    # Instantiate empty dictionary for this trial
                    trial_dict['logicRule'].append(logicRule)
                    trial_dict['sensoryRule'].append(sensoryRule)
                    trial_dict['motorRule'].append(motorRule)
                    trial_dict['stim1'].append(stim1)
                    trial_dict['stim2'].append(stim2)
                    trial_dict['motorResp'].append(motorResp)

    df = pd.DataFrame(trial_dict)
    df.to_csv(filename)

def constructTasksForRITLGeneralization(n_stims=1,filename_training='./GroupfMRI14a_EmpiricalSRActFlow_TrainingTasks.csv',
                                        filename_testing='./GroupfMRI14a_EmpiricalSRActFlow_TestTasks.csv'):
    """
    Construct and save a dictionary of tasks to simulate/generate data for
   
    """
    logicRules = ['BOTH', 'NOTBOTH', 'EITHER', 'NEITHER']
    sensoryRules = ['RED', 'VERTICAL', 'HIGH', 'CONSTANT']
    motorRules = ['LMID', 'LIND', 'RMID', 'RIND']
    colorStim = ['RED', 'BLUE']
    oriStim = ['VERTICAL', 'HORIZONTAL']
    pitchStim = ['HIGH', 'LOW']
    constantStim = ['CONSTANT','ALARM']

    #testset_tasks = ['BOTH-RED-LIND', 'NEITHER-VERTICAL-LMID', 'NOTBOTH-HIGH-RIND', 'EITHER-CONSTANT-RMID']
    #testset_tasks = ['BOTH-RED-LMID', 'NEITHER-VERTICAL-LIND', 'NOTBOTH-HIGH-RMID', 'EITHER-CONSTANT-RIND',
    #                 'NOTBOTH-RED-LMID', 'EITHER-VERTICAL-LIND', 'BOTH-HIGH-RMID', 'NEITHER-CONSTANT-RIND']
    testset_tasks = ['BOTH-RED-LMID', 'NEITHER-VERTICAL-LIND', 'NOTBOTH-HIGH-RMID', 'EITHER-CONSTANT-RIND',
                     'NOTBOTH-RED-LMID', 'EITHER-VERTICAL-LIND', 'BOTH-HIGH-RMID', 'NEITHER-CONSTANT-RIND',
                     'BOTH-VERTICAL-LMID', 'NEITHER-RED-LIND', 'NOTBOTH-CONSTANT-RMID', 'EITHER-HIGH-RIND',
                     'NOTBOTH-VERTICAL-LMID', 'EITHER-RED-LIND', 'BOTH-CONSTANT-RMID', 'NEITHER-HIGH-RIND',
                     'BOTH-CONSTANT-LMID', 'NEITHER-HIGH-LIND', 'NOTBOTH-VERTICAL-RMID', 'EITHER-RED-RIND',
                     'NOTBOTH-CONSTANT-LMID', 'EITHER-HIGH-LIND', 'BOTH-VERTICAL-RMID', 'NEITHER-RED-RIND',
                     'BOTH-HIGH-LMID', 'NEITHER-CONSTANT-LIND', 'NOTBOTH-RED-RMID', 'EITHER-VERTICAL-RIND',
                     'NOTBOTH-HIGH-LMID', 'EITHER-CONSTANT-LIND', 'BOTH-RED-RMID', 'NEITHER-VERTICAL-RIND']
#    testset_tasks = ['BOTH-RED-LMID', 'NEITHER-VERTICAL-LIND', 'NOTBOTH-HIGH-RMID', 'EITHER-CONSTANT-RIND',
#                     'NOTBOTH-RED-LMID', 'EITHER-VERTICAL-LIND', 'BOTH-HIGH-RMID', 'NEITHER-CONSTANT-RIND',
#                     'BOTH-VERTICAL-LMID', 'NEITHER-RED-LIND', 'NOTBOTH-CONSTANT-RMID', 'EITHER-HIGH-RIND',
#                     'NOTBOTH-VERTICAL-LMID', 'EITHER-RED-LIND', 'BOTH-CONSTANT-RMID', 'NEITHER-HIGH-RIND',
#                     'BOTH-CONSTANT-LMID', 'NEITHER-HIGH-LIND', 'NOTBOTH-VERTICAL-RMID', 'EITHER-RED-RIND',
#                     'NOTBOTH-CONSTANT-LMID', 'EITHER-HIGH-LIND', 'BOTH-VERTICAL-RMID', 'NEITHER-RED-RIND']

    # Training set dictionary
    trial_dict = {}
    trial_dict['logicRule'] = []
    trial_dict['sensoryRule'] = []
    trial_dict['motorRule'] = []
    trial_dict['stim1'] =  []
    trial_dict['stim2'] = []
    trial_dict['motorResp'] = []
    # Test set dictionary
    test_dict = {}
    test_dict['logicRule'] = []
    test_dict['sensoryRule'] = []
    test_dict['motorRule'] = []
    test_dict['stim1'] =  []
    test_dict['stim2'] = []
    test_dict['motorResp'] = []
    for sensoryRule in sensoryRules:

        for nstim in range(n_stims):
            # Randomly sample two stimulus patterns depending on the sensory rule
            if sensoryRule=='RED': stims = np.random.choice(colorStim,2,replace=True)
            if sensoryRule=='VERTICAL': stims = np.random.choice(oriStim,2,replace=True)
            if sensoryRule=='HIGH': stims = np.random.choice(pitchStim,2,replace=True)
            if sensoryRule=='PITCH': stims = np.random.choice(constantStim,2,replace=True)
            stim1, stim2 = stims

            
            for logicRule in logicRules:

                for motorRule in motorRules:

#                        if verbose:
#                            print 'Running actflow predictions for:', logicRule, sensoryRule, motorRule, 'task'

                    logicKey = 'RuleLogic_' + logicRule
                    sensoryKey = 'RuleSensory_' + sensoryRule
                    motorKey = 'RuleMotor_' + motorRule
                    stimKey = 'Stim_' + stim1 + stim2
                    motorResp = solveInputs(logicRule, sensoryRule, motorRule, stim1, stim2, printTask=False)
                    respKey = 'Response_' + motorResp
                    
                    task_str = logicRule + '-' + sensoryRule + '-' + motorRule
                    if task_str in testset_tasks:
                        test_dict['logicRule'].append(logicRule)
                        test_dict['sensoryRule'].append(sensoryRule)
                        test_dict['motorRule'].append(motorRule)
                        test_dict['stim1'].append(stim1)
                        test_dict['stim2'].append(stim2)
                        test_dict['motorResp'].append(motorResp)
                    else:
                        # Instantiate empty dictionary for this trial
                        trial_dict['logicRule'].append(logicRule)
                        trial_dict['sensoryRule'].append(sensoryRule)
                        trial_dict['motorRule'].append(motorRule)
                        trial_dict['stim1'].append(stim1)
                        trial_dict['stim2'].append(stim2)
                        trial_dict['motorResp'].append(motorResp)

    df = pd.DataFrame(trial_dict)
    df.to_csv(filename_training)
    df = pd.DataFrame(test_dict)
    df.to_csv(filename_testing)


