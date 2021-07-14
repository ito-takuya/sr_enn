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
import tools_group
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
        self.betas = h5f['taskRegression/ALL_24pXaCompCorXVolterra_taskReg_betas_canonical'][:].copy()
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
        srKey = 'sr' + sensoryrule + '_' + logicrule 
        stimKey = 'Stim_' + stim1 + stim2
        motorResp = solveInputs(logicrule, sensoryrule, motorrule, stim1, stim2, printTask=False)
        respKey = 'Response_' + motorResp


        stimKey_ind = np.where(self.stimCond==stimKey)[0]
        srKey_ind = np.where(self.stimCond==srKey)[0]
        motorKey_ind = np.where(self.stimCond==motorKey)[0]
        respKey_ind = np.where(self.stimCond==respKey)[0]


        stimData = self.betas[:,stimKey_ind].copy()
        srData = self.betas[:,srKey_ind].copy()
        motorRuleData = self.betas[:,motorKey_ind].copy()
        respData = self.betas[:,respKey_ind].copy()

        return motorResp, stimData, srData, motorRuleData, respData

    def generateActFlowPredictions(self,verbose=False,n_inputs=10):
        """
        Run all predictions for all 64 tasks
        """

        target_vertices = self.fc_motorrule2resp.shape[1]
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
                
                # Get input and sr indices for this sensory rule
                input_ind, sr_ind = self.getInputAndSRVertexInd(sensoryRule)

                for logicRule in self.logicRules:

                    for motorRule in self.motorRules:
                        if verbose:
                            print 'Running actflow predictions for:', logicRule, sensoryRule, motorRule, 'task'
                        # Get motor rule ind for this motor rule
                        motorrule_ind = self.getMotorRuleVertexInd(motorRule)

                        # Extract activations
                        motorResp, stimData, srData, motorRuleData, respData = self.extractActivations(logicRule, sensoryRule, motorRule, stims[0], stims[1])
                        respIndex.append(motorResp)

                        # Run activity flow

                        #### Input to SR regions
                        # first identify non-overlapping indices
                        unique_input_ind = np.where(np.in1d(input_ind,sr_ind)==False)[0]
                        #fc = self.fc_input2sr[sensoryRule]
                        fc = np.multiply(self.fc_input2sr[sensoryRule]>0,self.fc_input2sr[sensoryRule])
                        actflow_sr = np.dot(stimData[unique_input_ind,0],fc) 
                        
                        #### Construct composition sr representations
                        unique_sr_ind = np.where(np.in1d(sr_ind,motorrule_ind)==False)[0]
                        #sr_composition = actflow_sr + srData[sr_ind,0]
                        #sr_composition = np.multiply(np.multiply(actflow_sr>0,actflow_sr), np.multiply(srData[sr_ind,0]>0,srData[sr_ind,0])) 
                        sr_composition = np.multiply(np.multiply(actflow_sr>0,actflow_sr), srData[sr_ind,0])
                        #sr_composition = np.multiply(actflow_sr>0,actflow_sr) + srData[sr_ind,0]
                        
                        # Actflow
                        fc = np.multiply(self.fc_sr2motorrule[sensoryRule]>0,self.fc_sr2motorrule[sensoryRule])
                        #fc = self.fc_sr2motorrule[sensoryRule]
                        actflow_motorrule = np.dot(sr_composition[unique_sr_ind],fc)
                        
                        #### Construct motor rule compositions
                        unique_motor_ind = np.where(np.in1d(motorrule_ind,self.target_ind)==False)[0]
                        #motorrule_composition = actflow_motorrule + motorRuleData[motorrule_ind,0] 
                        #motorrule_composition = np.multiply(np.multiply(actflow_motorrule>0,actflow_motorrule), np.multiply(motorRuleData[motorrule_ind,0]>0,motorRuleData[motorrule_ind,0]))
                        motorrule_composition = np.multiply(np.multiply(actflow_motorrule>0,actflow_motorrule), motorRuleData[motorrule_ind,0])
                        #motorrule_composition = np.multiply(actflow_motorrule>0,actflow_motorrule) + motorRuleData[motorrule_ind,0]

                        # Actflow
                        fc = np.multiply(self.fc_motorrule2resp>0,self.fc_motorrule2resp)
                        #fc = self.fc_motorrule2resp
                        actflow_predictions.append(np.dot(motorrule_composition[unique_motor_ind],fc))


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

    def getInputAndSRVertexInd(self,sensoryRule):
        if sensoryRule=='RED': 
            inputkey = 'COLOR'
            srkey =  'srRed'
        if sensoryRule=='VERTICAL':
            inputkey = 'ORI'
            srkey = 'srVertical'
        if sensoryRule=='HIGH':
            inputkey = 'PITCH'
            srkey = 'srHigh'
        if sensoryRule=='CONSTANT': 
            inputkey = 'CONSTANT'
            srkey = 'srConstant'
        
        inputdir = '/projects3/SRActFlow/data/results/GroupfMRI/InputStimuliDecoding/'
        input_regions = np.loadtxt(inputdir + 'InputStimuliRegions_' + inputkey + '.csv',delimiter=',')
        srdir = '/projects3/SRActFlow/data/results/GroupfMRI/SRDecoding/'
        sr_regions = np.loadtxt(srdir + srkey + '_Regions.csv',delimiter=',')

        input_ind = []
        for roi in input_regions:
            input_ind.extend(np.where(glasser2==roi+1)[0])
        sr_ind = []
        for roi in sr_regions:
            sr_ind.extend(np.where(glasser2==roi+1)[0])
        
        return input_ind, sr_ind

    def getMotorRuleVertexInd(self,motorRule):
        ruledir = '/projects3/SRActFlow/data/results/GroupfMRI/RuleDecoding/'
        motor_regions = np.loadtxt(ruledir + 'MotorRule_Regions.csv',delimiter=',')
        motorrule_ind = []
        for roi in motor_regions:
            motorrule_ind.extend(np.where(glasser2==roi+1)[0])

        return motorrule_ind


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
