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
import sklearn.svm as svm
import statsmodels.sandbox.stats.multicomp as mc
import sklearn
from sklearn.feature_selection import f_classif
import h5py
import glmScripts.taskGLMPipeline_v2 as tgp
import sys
import utils.loadExperimentalData as led
import tools_group_rsa
tgp = reload(tgp)

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

class EmpiricalActFlow():
    """
    Class to perform empirical actflow for a given subject (stimulus-to-response)
    """
    def __init__(self,subj):
        """
        instantiate:
            indices for condition types
            indices for specific condition instances
            betas
        """
        self.subj = subj

        X = tgp.loadTaskTiming(subj,'ALL')
        self.stimIndex = np.asarray(X['stimIndex'])
        self.stimCond = np.asarray(X['stimCond'])

        datadir = basedir + 'data/postProcessing/hcpPostProcCiric/'
        h5f = h5py.File(datadir + subj + '_glmOutput_data.h5','r')
        self.betas = np.real(h5f['taskRegression/ALL_24pXaCompCorXVolterra_taskReg_betas_canonical'][:].copy())
        h5f.close()

        self.logicRules = ['BOTH', 'NOTBOTH', 'EITHER', 'NEITHER']
        self.sensoryRules = ['RED', 'VERTICAL', 'HIGH', 'CONSTANT']
        self.motorRules = ['LMID', 'LIND', 'RMID', 'RIND']
        self.colorStim = ['RED', 'BLUE']
        self.oriStim = ['VERTICAL', 'HORIZONTAL']
        self.pitchStim = ['HIGH', 'LOW']
        self.constantStim = ['CONSTANT','ALARM']

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
        self.target_ind = target_ind

    def extractActivations(self, logicrule, sensoryrule, motorrule, stim1, stim2):
        """
        extract activations, including motor response
        """
        logicKey = 'RuleLogic_' + logicrule
        sensoryKey = 'RuleSensory_' + sensoryrule
        motorKey = 'RuleMotor_' + motorrule
        stimKey = 'Stim_' + stim1 + stim2
        motorResp = solveInputs(logicrule, sensoryrule, motorrule, stim1, stim2, printTask=False)
        respKey = 'Response_' + motorResp


        stimKey_ind = np.where(self.stimCond==stimKey)[0]
        logicRule_ind = np.where(self.stimCond==logicKey)[0]
        sensoryRule_ind = np.where(self.stimCond==sensoryKey)[0]
        motorRule_ind = np.where(self.stimCond==motorKey)[0]
        respKey_ind = np.where(self.stimCond==respKey)[0]


        stimData = self.betas[:,stimKey_ind].copy()
        logicRuleData = self.betas[:,logicRule_ind].copy()
        sensoryRuleData = self.betas[:,sensoryRule_ind].copy()
        motorRuleData = self.betas[:,motorRule_ind].copy()
        respData = self.betas[:,respKey_ind].copy()

        return motorResp, stimData, logicRuleData, sensoryRuleData, motorRuleData, respData

    def generateActFlowPredictions(self,verbose=False,n_inputs=10):
        """
        Run all predictions for all 64 tasks
        """

        target_vertices = self.fc_hidden2motorresp.shape[1]
        actflow = np.zeros((target_vertices,4,n_inputs)) #LMID, LIND, RMID, rIND
        for nstim in range(n_inputs):
            respIndex = []
            actflow_predictions = []
            for sensoryRule in self.sensoryRules:
                # Randomly sample two stimulus patterns depending on the sensory rule
                if sensoryRule=='RED': stims = np.random.choice(self.colorStim,2,replace=True)
                if sensoryRule=='VERTICAL': stims = np.random.choice(self.oriStim,2,replace=True)
                if sensoryRule=='HIGH': stims = np.random.choice(self.pitchStim,2,replace=True)
                if sensoryRule=='PITCH': stims = np.random.choice(self.constantStim,2,replace=True)
                

                for logicRule in self.logicRules:

                    for motorRule in self.motorRules:
                        # Get input, rule, and hidden indices
                        input_ind, logic_ind, sensory_ind, motor_ind, hidden_ind = self.getRoiIndices(logicRule,sensoryRule,motorRule)

                        if verbose:
                            print 'Running actflow predictions for:', logicRule, sensoryRule, motorRule, 'task'

                        # Extract activations
                        motorResp, stimData, logicRuleData, sensoryRuleData, motorRuleData, respData = self.extractActivations(logicRule, sensoryRule, motorRule, stims[0], stims[1])
                        respIndex.append(motorResp)

                        # Run activity flow

                        #### Input to hidden regions
                        # first identify non-overlapping indices
                        unique_input_ind = np.where(np.in1d(input_ind,hidden_ind)==False)[0]
                        fc = self.fc_input2hidden[sensoryRule]
                        #print('Size of stimdata ' + str(stimData[unique_input_ind,:].shape))
                        actflow_stim = np.dot(stimData[unique_input_ind,0],fc) 
                        
                        #### Logic rule to hidden regions
                        # first identify non-overlapping indices
                        unique_input_ind = np.where(np.in1d(logic_ind,hidden_ind)==False)[0]
                        fc = self.fc_logic2hidden
                        #print('Size of logicdata ' + str(logicRuleData[unique_input_ind,:].shape))
                        actflow_logicrule = np.dot(logicRuleData[unique_input_ind,0],fc) 

                        #### Sensory rule to hidden regions
                        # first identify non-overlapping indices
                        unique_input_ind = np.where(np.in1d(sensory_ind,hidden_ind)==False)[0]
                        fc = self.fc_sensory2hidden
                        #print('Size of sensorydata ' + str(sensoryRuleData[unique_input_ind,:].shape))
                        actflow_sensoryrule = np.dot(sensoryRuleData[unique_input_ind,0],fc) 
                        
                        #### Motor rule to hidden regions
                        # first identify non-overlapping indices
                        unique_input_ind = np.where(np.in1d(motor_ind,hidden_ind)==False)[0]
                        fc = self.fc_motor2hidden
                        #print('Size of motordata ' + str(motorRuleData[unique_input_ind,:].shape))
                        actflow_motorrule = np.dot(motorRuleData[unique_input_ind,0],fc) 

                        #### Compositionality
                        ## Compositional representations in hidden layers
                        thresh = 0
                        #actflow_stim = np.multiply(actflow_stim>thresh,actflow_stim)
                        #actflow_logicrule = np.multiply(actflow_logicrule>thresh,actflow_logicrule)
                        #actflow_sensoryrule = np.multiply(actflow_sensoryrule>thresh,actflow_sensoryrule)
                        #actflow_motorrule = np.multiply(actflow_motorrule>thresh,actflow_motorrule)
                        #hiddenlayer_composition = actflow_stim + actflow_logicrule + actflow_sensoryrule + actflow_motorrule
                        # multiplicative gating
                        hiddenlayer_composition = np.multiply(np.multiply(np.multiply(actflow_stim, actflow_logicrule), actflow_sensoryrule), actflow_motorrule)

                        #### Hidden to output regions 
                        unique_ind = np.where(np.in1d(hidden_ind,self.target_ind)==False)[0]
                        fc = self.fc_hidden2motorresp
                        actflow_predictions.append(np.dot(hiddenlayer_composition[unique_ind],fc))


            respIndex = np.asarray(respIndex)
            actflow_predictions = np.asarray(actflow_predictions)
            # LMID
            ind = np.where(respIndex=='LMID')[0]
            actflow[:,0,nstim] = np.mean(actflow_predictions[ind,:],axis=0)
            # LIND
            ind = np.where(respIndex=='LIND')[0]
            actflow[:,1,nstim] = np.mean(actflow_predictions[ind,:],axis=0)
            # RMID
            ind = np.where(respIndex=='RMID')[0]
            actflow[:,2,nstim] = np.mean(actflow_predictions[ind,:],axis=0)
            # RIND
            ind = np.where(respIndex=='RIND')[0]
            actflow[:,3,nstim] = np.mean(actflow_predictions[ind,:],axis=0)

        actflow = np.mean(actflow,axis=2)

        return actflow

    def generateActFlowPredictionsControl_NoTaskContext(self,verbose=False,n_inputs=10):
        """
        Run all predictions for all 64 tasks
        """

        target_vertices = self.fc_hidden2motorresp.shape[1]
        actflow = np.zeros((target_vertices,4,n_inputs)) #LMID, LIND, RMID, rIND
        for nstim in range(n_inputs):
            respIndex = []
            actflow_predictions = []
            for sensoryRule in self.sensoryRules:
                # Randomly sample two stimulus patterns depending on the sensory rule
                if sensoryRule=='RED': stims = np.random.choice(self.colorStim,2,replace=True)
                if sensoryRule=='VERTICAL': stims = np.random.choice(self.oriStim,2,replace=True)
                if sensoryRule=='HIGH': stims = np.random.choice(self.pitchStim,2,replace=True)
                if sensoryRule=='PITCH': stims = np.random.choice(self.constantStim,2,replace=True)
                

                for logicRule in self.logicRules:

                    for motorRule in self.motorRules:
                        # Get input, rule, and hidden indices
                        input_ind, logic_ind, sensory_ind, motor_ind, hidden_ind = self.getRoiIndices(logicRule,sensoryRule,motorRule)

                        if verbose:
                            print 'Running actflow predictions for:', logicRule, sensoryRule, motorRule, 'task'

                        # Extract activations
                        motorResp, stimData, logicRuleData, sensoryRuleData, motorRuleData, respData = self.extractActivations(logicRule, sensoryRule, motorRule, stims[0], stims[1])
                        respIndex.append(motorResp)

                        # Run activity flow

                        #### Input to hidden regions
                        # first identify non-overlapping indices
                        unique_input_ind = np.where(np.in1d(input_ind,hidden_ind)==False)[0]
                        fc = self.fc_input2hidden[sensoryRule]
                        #print('Size of stimdata ' + str(stimData[unique_input_ind,:].shape))
                        actflow_stim = np.dot(stimData[unique_input_ind,0],fc) 
                        

                        #### Compositionality
                        ## Compositional representations in hidden layers
                        thresh = 0
                        #actflow_stim = np.multiply(actflow_stim>thresh,actflow_stim)
                        #actflow_logicrule = np.multiply(actflow_logicrule>thresh,actflow_logicrule)
                        #actflow_sensoryrule = np.multiply(actflow_sensoryrule>thresh,actflow_sensoryrule)
                        #actflow_motorrule = np.multiply(actflow_motorrule>thresh,actflow_motorrule)
                        #hiddenlayer_composition = actflow_stim + actflow_logicrule + actflow_sensoryrule + actflow_motorrule
                        # multiplicative gating
                        #hiddenlayer_composition = np.multiply(np.multiply(np.multiply(actflow_stim, actflow_logicrule), actflow_sensoryrule), actflow_motorrule)
                        hiddenlayer_composition = actflow_stim 

                        #### Hidden to output regions 
                        unique_ind = np.where(np.in1d(hidden_ind,self.target_ind)==False)[0]
                        fc = self.fc_hidden2motorresp
                        actflow_predictions.append(np.dot(hiddenlayer_composition[unique_ind],fc))


            respIndex = np.asarray(respIndex)
            actflow_predictions = np.asarray(actflow_predictions)
            # LMID
            ind = np.where(respIndex=='LMID')[0]
            actflow[:,0,nstim] = np.mean(actflow_predictions[ind,:],axis=0)
            # LIND
            ind = np.where(respIndex=='LIND')[0]
            actflow[:,1,nstim] = np.mean(actflow_predictions[ind,:],axis=0)
            # RMID
            ind = np.where(respIndex=='RMID')[0]
            actflow[:,2,nstim] = np.mean(actflow_predictions[ind,:],axis=0)
            # RIND
            ind = np.where(respIndex=='RIND')[0]
            actflow[:,3,nstim] = np.mean(actflow_predictions[ind,:],axis=0)

        actflow = np.mean(actflow,axis=2)

        return actflow

    def generateActFlowPredictionsControl_NoSensoryStim(self,verbose=False,n_inputs=10):
        """
        Run all predictions for all 64 tasks
        """

        target_vertices = self.fc_hidden2motorresp.shape[1]
        actflow = np.zeros((target_vertices,4,n_inputs)) #LMID, LIND, RMID, rIND
        for nstim in range(n_inputs):
            respIndex = []
            actflow_predictions = []
            for sensoryRule in self.sensoryRules:
                # Randomly sample two stimulus patterns depending on the sensory rule
                if sensoryRule=='RED': stims = np.random.choice(self.colorStim,2,replace=True)
                if sensoryRule=='VERTICAL': stims = np.random.choice(self.oriStim,2,replace=True)
                if sensoryRule=='HIGH': stims = np.random.choice(self.pitchStim,2,replace=True)
                if sensoryRule=='PITCH': stims = np.random.choice(self.constantStim,2,replace=True)
                

                for logicRule in self.logicRules:

                    for motorRule in self.motorRules:
                        # Get input, rule, and hidden indices
                        input_ind, logic_ind, sensory_ind, motor_ind, hidden_ind = self.getRoiIndices(logicRule,sensoryRule,motorRule)

                        if verbose:
                            print 'Running actflow predictions for:', logicRule, sensoryRule, motorRule, 'task'

                        # Extract activations
                        motorResp, stimData, logicRuleData, sensoryRuleData, motorRuleData, respData = self.extractActivations(logicRule, sensoryRule, motorRule, stims[0], stims[1])
                        respIndex.append(motorResp)

                        # Run activity flow

                        #### Logic rule to hidden regions
                        # first identify non-overlapping indices
                        unique_input_ind = np.where(np.in1d(logic_ind,hidden_ind)==False)[0]
                        fc = self.fc_logic2hidden
                        #print('Size of logicdata ' + str(logicRuleData[unique_input_ind,:].shape))
                        actflow_logicrule = np.dot(logicRuleData[unique_input_ind,0],fc) 

                        #### Sensory rule to hidden regions
                        # first identify non-overlapping indices
                        unique_input_ind = np.where(np.in1d(sensory_ind,hidden_ind)==False)[0]
                        fc = self.fc_sensory2hidden
                        #print('Size of sensorydata ' + str(sensoryRuleData[unique_input_ind,:].shape))
                        actflow_sensoryrule = np.dot(sensoryRuleData[unique_input_ind,0],fc) 
                        
                        #### Motor rule to hidden regions
                        # first identify non-overlapping indices
                        unique_input_ind = np.where(np.in1d(motor_ind,hidden_ind)==False)[0]
                        fc = self.fc_motor2hidden
                        #print('Size of motordata ' + str(motorRuleData[unique_input_ind,:].shape))
                        actflow_motorrule = np.dot(motorRuleData[unique_input_ind,0],fc) 

                        #### Compositionality
                        ## Compositional representations in hidden layers
                        thresh = 0
                        #actflow_stim = np.multiply(actflow_stim>thresh,actflow_stim)
                        #actflow_logicrule = np.multiply(actflow_logicrule>thresh,actflow_logicrule)
                        #actflow_sensoryrule = np.multiply(actflow_sensoryrule>thresh,actflow_sensoryrule)
                        #actflow_motorrule = np.multiply(actflow_motorrule>thresh,actflow_motorrule)
                        #hiddenlayer_composition = actflow_stim + actflow_logicrule + actflow_sensoryrule + actflow_motorrule
                        # multiplicative gating
                        hiddenlayer_composition = np.multiply(np.multiply(actflow_logicrule, actflow_sensoryrule), actflow_motorrule)

                        #### Hidden to output regions 
                        unique_ind = np.where(np.in1d(hidden_ind,self.target_ind)==False)[0]
                        fc = self.fc_hidden2motorresp
                        actflow_predictions.append(np.dot(hiddenlayer_composition[unique_ind],fc))


            respIndex = np.asarray(respIndex)
            actflow_predictions = np.asarray(actflow_predictions)
            # LMID
            ind = np.where(respIndex=='LMID')[0]
            actflow[:,0,nstim] = np.mean(actflow_predictions[ind,:],axis=0)
            # LIND
            ind = np.where(respIndex=='LIND')[0]
            actflow[:,1,nstim] = np.mean(actflow_predictions[ind,:],axis=0)
            # RMID
            ind = np.where(respIndex=='RMID')[0]
            actflow[:,2,nstim] = np.mean(actflow_predictions[ind,:],axis=0)
            # RIND
            ind = np.where(respIndex=='RIND')[0]
            actflow[:,3,nstim] = np.mean(actflow_predictions[ind,:],axis=0)

        actflow = np.mean(actflow,axis=2)

        return actflow

    def getRoiIndices(self,logicRule,sensoryRule,motorRule,n_hiddenregions=10):
        if sensoryRule=='RED': 
            inputkey = 'COLOR'
        if sensoryRule=='VERTICAL':
            inputkey = 'ORI'
        if sensoryRule=='HIGH':
            inputkey = 'PITCH'
        if sensoryRule=='CONSTANT': 
            inputkey = 'CONSTANT'
        
        inputdir = '/projects3/SRActFlow/data/results/GroupfMRI/InputStimuliDecoding/'
        input_regions = np.loadtxt(inputdir + 'InputStimuliRegions_' + inputkey + '.csv',delimiter=',')
        
        ruledir = '/projects3/SRActFlow/data/results/GroupfMRI/RuleDecoding/'
        logic_regions = np.loadtxt(ruledir + 'LogicRule_Regions.csv',delimiter=',')
        sensory_regions = np.loadtxt(ruledir + 'SensoryRule_Regions.csv',delimiter=',')
        motor_regions = np.loadtxt(ruledir + 'MotorRule_Regions.csv',delimiter=',')

        hiddendir = '/projects3/SRActFlow/data/results/GroupfMRI/RSA/'
        hiddenregions = np.loadtxt(hiddendir + 'RSA_Similarity_SortedRegions.txt',delimiter=',')
        hiddenregions = hiddenregions[:n_hiddenregions]

        input_ind = []
        for roi in input_regions:
            input_ind.extend(np.where(glasser2==roi+1)[0])
        logic_ind = []
        for roi in logic_regions:
            logic_ind.extend(np.where(glasser2==roi+1)[0])
        sensory_ind = []
        for roi in sensory_regions:
            sensory_ind.extend(np.where(glasser2==roi+1)[0])
        motor_ind = []
        for roi in motor_regions:
            motor_ind.extend(np.where(glasser2==roi+1)[0])
        hidden_ind = []
        for roi in hiddenregions:
            hidden_ind.extend(np.where(glasser2==roi+1)[0])
        
        return input_ind, logic_ind, sensory_ind, motor_ind, hidden_ind


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
        print 'Logic rule:', logicRule
        print 'Sensory rule:', sensoryRule
        print 'Motor rule:', motorRule
        print '**Stimuli**'
        print stim1, stim2
        print 'Motor response:', motorOutput

    return motorOutput
