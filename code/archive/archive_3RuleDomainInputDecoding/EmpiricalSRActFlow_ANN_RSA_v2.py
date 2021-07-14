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
import pandas as pd

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

glasserfile2 = '/projects3/AnalysisTools/ParcelsGlasser2016/Q1-Q6_RelatedParcellation210.LR.CorticalAreas_dil_Colors.32k_fs_RL.dlabel.nii'
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

    def extractAllActivations(self, df_trials):
        """
        extract activations, including motor response
        """
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

    def generateActFlowPredictions(self,verbose=False):
        """
        Run all predictions for all 64 tasks
        """

        target_vertices = self.fc_hidden2motorresp.shape[1]
        actflow = np.zeros((target_vertices,4,4)) #LMID, LIND, RMID, rIND -- 4 cols in 3rd dim for each sensory rule
        sensecount = 0
        for sensoryRule in self.sensoryRules:
            sensoryIndices = np.where(np.asarray(self.sensoryRuleIndices)==sensoryRule)[0]

            input_ind, logic_ind, sensory_ind, motor_ind, hidden_ind = self.getRoiIndices(sensoryRule) # Can randomly choose logic/motor rules - doesn't maek a difference. need to edit this

            # Run activity flow

            #### Input to hidden regions
            # first identify non-overlapping indices
            unique_input_ind = np.where(np.in1d(input_ind,hidden_ind)==False)[0]
            fc = self.fc_input2hidden[sensoryRule]
            #print('Size of stimdata ' + str(stimData[unique_input_ind,:].shape))
            actflow_stim = np.matmul(self.stimData[:,unique_input_ind],fc) 
            
            #### Logic rule to hidden regions
            # first identify non-overlapping indices
            unique_input_ind = np.where(np.in1d(logic_ind,hidden_ind)==False)[0]
            fc = self.fc_logic2hidden
            #print('Size of logicdata ' + str(logicRuleData[unique_input_ind,:].shape))
            actflow_logicrule = np.matmul(self.logicRuleData[:,unique_input_ind],fc) 

            #### Sensory rule to hidden regions
            # first identify non-overlapping indices
            unique_input_ind = np.where(np.in1d(sensory_ind,hidden_ind)==False)[0]
            fc = self.fc_sensory2hidden
            #print('Size of sensorydata ' + str(sensoryRuleData[unique_input_ind,:].shape))
            actflow_sensoryrule = np.matmul(self.sensoryRuleData[:,unique_input_ind],fc) 
            
            #### Motor rule to hidden regions
            # first identify non-overlapping indices
            unique_input_ind = np.where(np.in1d(motor_ind,hidden_ind)==False)[0]
            fc = self.fc_motor2hidden
            #print('Size of motordata ' + str(motorRuleData[unique_input_ind,:].shape))
            actflow_motorrule = np.matmul(self.motorRuleData[:,unique_input_ind],fc) 

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
            actflow_predictions = np.matmul(hiddenlayer_composition[:,unique_ind],fc)


            respIndex = np.asarray(self.motorRespAll)[sensoryIndices]
            # LMID
            ind = np.where(respIndex=='LMID')[0]
            actflow[:,0,sensecount] = np.mean(actflow_predictions[ind,:],axis=0)
            # LIND
            ind = np.where(respIndex=='LIND')[0]
            actflow[:,1,sensecount] = np.mean(actflow_predictions[ind,:],axis=0)
            # RMID
            ind = np.where(respIndex=='RMID')[0]
            actflow[:,2,sensecount] = np.mean(actflow_predictions[ind,:],axis=0)
            # RIND
            ind = np.where(respIndex=='RIND')[0]
            actflow[:,3,sensecount] = np.mean(actflow_predictions[ind,:],axis=0)

            sensecount += 1

        actflow = np.mean(actflow,axis=2)

        return actflow

    def generateActFlowPredictions_PCFC(self,thresh=0,verbose=False):
        """
        Run all predictions for all 64 tasks
        """
        ### Threshold
        #if thresh==None:
        #    pass
        #else:
        #    self.logicRuleData = np.multiply(self.logicRuleData,self.logicRuleData>thresh)
        #    self.sensoryRuleData = np.multiply(self.sensoryRuleData,self.sensoryRuleData>thresh)
        #    self.motorRuleData = np.multiply(self.motorRuleData,self.motorRuleData>thresh)

        target_vertices = self.fc_hidden2motorresp.shape[1]
        actflow = np.zeros((target_vertices,4)) #LMID, LIND, RMID, rIND -- 4 cols in 3rd dim for each sensory rule
        sensecount = 0
        for sensoryRule in self.sensoryRules:
            sensoryIndices = np.where(np.asarray(self.sensoryRuleIndices)==sensoryRule)[0]

            input_ind, logic_ind, sensory_ind, motor_ind, hidden_ind = self.getRoiIndices(sensoryRule) # Can randomly choose logic/motor rules - doesn't maek a difference. need to edit this

            # Run activity flow

            #### Input to hidden regions
            # first identify non-overlapping indices
            unique_input_ind = np.where(np.in1d(input_ind,hidden_ind)==False)[0]
            fc = self.fc_input2hidden[sensoryRule]
            pc_act = np.matmul(self.stimData[sensoryIndices,:][:,unique_input_ind],self.eig_input2hidden[sensoryRule].T)
            actflow_stim = np.matmul(pc_act,fc) 
            
            #### Logic rule to hidden regions
            # first identify non-overlapping indices
            unique_input_ind = np.where(np.in1d(logic_ind,hidden_ind)==False)[0]
            fc = self.fc_logic2hidden
            pc_act = np.matmul(self.logicRuleData[sensoryIndices,:][:,unique_input_ind],self.eig_logic2hidden.T)
            actflow_logicrule = np.matmul(pc_act,fc) 

            #### Sensory rule to hidden regions
            # first identify non-overlapping indices
            unique_input_ind = np.where(np.in1d(sensory_ind,hidden_ind)==False)[0]
            fc = self.fc_sensory2hidden
            pc_act = np.matmul(self.sensoryRuleData[sensoryIndices,:][:,unique_input_ind],self.eig_sensory2hidden.T)
            actflow_sensoryrule = np.matmul(pc_act,fc) 
            
            #### Motor rule to hidden regions
            # first identify non-overlapping indices
            unique_input_ind = np.where(np.in1d(motor_ind,hidden_ind)==False)[0]
            fc = self.fc_motor2hidden
            pc_act = np.matmul(self.motorRuleData[sensoryIndices,:][:,unique_input_ind],self.eig_motor2hidden.T)
            actflow_motorrule = np.matmul(pc_act,fc) 

            #### Compositionality
            #actflow_logicrule = np.multiply(actflow_logicrule>thresh,actflow_logicrule)
            #actflow_sensoryrule = np.multiply(actflow_sensoryrule>thresh,actflow_sensoryrule)
            #actflow_motorrule = np.multiply(actflow_motorrule>thresh,actflow_motorrule)
            ### Compositional representations in hidden layers
            #hiddenlayer_composition = actflow_stim + actflow_logicrule + actflow_sensoryrule + actflow_motorrule

            actflow_taskrules = actflow_logicrule + actflow_sensoryrule + actflow_motorrule
            if thresh!=None:
                actflow_taskrules = np.multiply(actflow_taskrules,actflow_taskrules>0)
                actflow_stim = np.multiply(actflow_stim,actflow_stim>0)
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
            unique_ind = np.where(np.in1d(hidden_ind,self.target_ind)==False)[0]
            fc = self.fc_hidden2motorresp
            pc_act = np.matmul(hiddenlayer_composition[:,unique_ind],self.eig_hidden2motorresp.T)
            actflow_predictions = np.real(np.matmul(pc_act,fc))


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

    def generateActFlowPredictions_PCFC_Control(self,control='stim',thresh=0,verbose=False):
        """
        Run all predictions for all 64 tasks
        """

        ### Threshold
        #if thresh==None:
        #    pass
        #else:
        #    self.logicRuleData = np.multiply(self.logicRuleData,self.logicRuleData>thresh)
        #    self.sensoryRuleData = np.multiply(self.sensoryRuleData,self.sensoryRuleData>thresh)
        #    self.motorRuleData = np.multiply(self.motorRuleData,self.motorRuleData>thresh)

        target_vertices = self.fc_hidden2motorresp.shape[1]
        actflow = np.zeros((target_vertices,4,4)) #LMID, LIND, RMID, rIND -- 4 cols in 3rd dim for each sensory rule
        sensecount = 0
        for sensoryRule in self.sensoryRules:
            sensoryIndices = np.where(np.asarray(self.sensoryRuleIndices)==sensoryRule)[0]

            input_ind, logic_ind, sensory_ind, motor_ind, hidden_ind = self.getRoiIndices(sensoryRule) # Can randomly choose logic/motor rules - doesn't maek a difference. need to edit this

            # Run activity flow

            #### Input to hidden regions
            # first identify non-overlapping indices
            unique_input_ind = np.where(np.in1d(input_ind,hidden_ind)==False)[0]
            fc = self.fc_input2hidden[sensoryRule]
            pc_act = np.matmul(self.stimData[sensoryIndices,:][:,unique_input_ind],self.eig_input2hidden[sensoryRule].T)
            actflow_stim = np.matmul(pc_act,fc) 
            
            #### Logic rule to hidden regions
            # first identify non-overlapping indices
            unique_input_ind = np.where(np.in1d(logic_ind,hidden_ind)==False)[0]
            fc = self.fc_logic2hidden
            pc_act = np.matmul(self.logicRuleData[sensoryIndices,:][:,unique_input_ind],self.eig_logic2hidden.T)
            actflow_logicrule = np.matmul(pc_act,fc) 

            #### Sensory rule to hidden regions
            # first identify non-overlapping indices
            unique_input_ind = np.where(np.in1d(sensory_ind,hidden_ind)==False)[0]
            fc = self.fc_sensory2hidden
            pc_act = np.matmul(self.sensoryRuleData[sensoryIndices,:][:,unique_input_ind],self.eig_sensory2hidden.T)
            actflow_sensoryrule = np.matmul(pc_act,fc) 
            
            #### Motor rule to hidden regions
            # first identify non-overlapping indices
            unique_input_ind = np.where(np.in1d(motor_ind,hidden_ind)==False)[0]
            fc = self.fc_motor2hidden
            pc_act = np.matmul(self.motorRuleData[sensoryIndices,:][:,unique_input_ind],self.eig_motor2hidden.T)
            actflow_motorrule = np.matmul(pc_act,fc) 

            #### Compositionality
            ## Compositional representations in hidden layers

            #actflow_stim = np.multiply(actflow_stim>thresh,actflow_stim)
            actflow_logicrule = np.multiply(actflow_logicrule>thresh,actflow_logicrule)
            actflow_sensoryrule = np.multiply(actflow_sensoryrule>thresh,actflow_sensoryrule)
            actflow_motorrule = np.multiply(actflow_motorrule>thresh,actflow_motorrule)

            if control=='stim':
                actflow_taskrules = actflow_logicrule + actflow_sensoryrule + actflow_motorrule
                if thresh!=None:
                    actflow_taskrules = np.multiply(actflow_taskrules,actflow_taskrules>0)
                    actflow_stim = np.multiply(actflow_stim,actflow_stim>0)
                hiddenlayer_composition = actflow_taskrules
            elif control=='logic':
                actflow_taskrules = actflow_sensoryrule + actflow_motorrule
                if thresh!=None:
                    actflow_taskrules = np.multiply(actflow_taskrules,actflow_taskrules>0)
                    actflow_stim = np.multiply(actflow_stim,actflow_stim>0)
                hiddenlayer_composition = actflow_taskrules + actflow_stim
            elif control=='sensory':
                actflow_taskrules = actflow_logicrule + actflow_motorrule
                if thresh!=None:
                    actflow_taskrules = np.multiply(actflow_taskrules,actflow_taskrules>0)
                    actflow_stim = np.multiply(actflow_stim,actflow_stim>0)
                hiddenlayer_composition = actflow_taskrules + actflow_stim
            elif control=='motor':
                actflow_taskrules = actflow_sensoryrule + actflow_logicrule
                if thresh!=None:
                    actflow_taskrules = np.multiply(actflow_taskrules,actflow_taskrules>0)
                    actflow_stim = np.multiply(actflow_stim,actflow_stim>0)
                hiddenlayer_composition = actflow_taskrules + actflow_stim

            if thresh!=None:
                hiddenlayer_composition = np.multiply(hiddenlayer_composition,hiddenlayer_composition>thresh)
               #t, p = stats.ttest_1samp(hiddenlayer_composition,0,axis=0) # Trials x Vertices
               #p[t>0] = p[t>0]/2.0
               #p[t<0] = 1.0 - p[t<0]/2.0
               #h0 = mc.fdrcorrection0(p)[0] # Effectively a threshold linear func
               #h0 = p<0.1
               #hiddenlayer_composition = np.multiply(hiddenlayer_composition,h0)

            # multiplicative gating
            #hiddenlayer_composition = np.multiply(np.multiply(actflow_logicrule, actflow_sensoryrule), actflow_motorrule)

            #### Hidden to output regions 
            unique_ind = np.where(np.in1d(hidden_ind,self.target_ind)==False)[0]
            fc = self.fc_hidden2motorresp
            pc_act = np.matmul(hiddenlayer_composition[:,unique_ind],self.eig_hidden2motorresp.T)
            actflow_predictions = np.real(np.matmul(pc_act,fc))


            respIndex = np.asarray(self.motorRespAll)[sensoryIndices]
            # LMID
            ind = np.where(respIndex=='LMID')[0]
            actflow[:,0,sensecount] = np.mean(actflow_predictions[ind,:],axis=0)
            # LIND
            ind = np.where(respIndex=='LIND')[0]
            actflow[:,1,sensecount] = np.mean(actflow_predictions[ind,:],axis=0)
            # RMID
            ind = np.where(respIndex=='RMID')[0]
            actflow[:,2,sensecount] = np.mean(actflow_predictions[ind,:],axis=0)
            # RIND
            ind = np.where(respIndex=='RIND')[0]
            actflow[:,3,sensecount] = np.mean(actflow_predictions[ind,:],axis=0)

            sensecount += 1

        actflow = np.mean(actflow,axis=2)

        return actflow

    def generateInputControlDecoding(self,verbose=False):
        """
        Run all predictions for all 64 tasks
        """
        ### Threshold
        #if thresh==None:
        #    pass
        #else:
        #    self.logicRuleData = np.multiply(self.logicRuleData,self.logicRuleData>thresh)
        #    self.sensoryRuleData = np.multiply(self.sensoryRuleData,self.sensoryRuleData>thresh)
        #    self.motorRuleData = np.multiply(self.motorRuleData,self.motorRuleData>thresh)

        target_vertices = self.fc_hidden2motorresp.shape[1]
        actflow = np.zeros((target_vertices,4)) #LMID, LIND, RMID, rIND -- 4 cols in 3rd dim for each sensory rule
        input_activations_lmid = []
        input_activations_lind = []
        input_activations_rmid = []
        input_activations_rind = []
        sensecount = 0
        for sensoryRule in self.sensoryRules:
            sensoryIndices = np.where(np.asarray(self.sensoryRuleIndices)==sensoryRule)[0]

            input_ind, logic_ind, sensory_ind, motor_ind, hidden_ind = self.getRoiIndices(sensoryRule) # Can randomly choose logic/motor rules - doesn't maek a difference. need to edit this

            # Run activity flow

            #### Input activations
            unique_input_ind = np.where(np.in1d(input_ind,hidden_ind)==False)[0]
            input_act = self.stimData[sensoryIndices,:][:,unique_input_ind]
            
            #### Logic rule activations
            unique_input_ind = np.where(np.in1d(logic_ind,hidden_ind)==False)[0]
            logic_act = self.logicRuleData[sensoryIndices,:][:,unique_input_ind]

            #### Sensory rule activations
            unique_input_ind = np.where(np.in1d(sensory_ind,hidden_ind)==False)[0]
            sensory_act = self.sensoryRuleData[sensoryIndices,:][:,unique_input_ind]
            
            #### Motor rule to hidden regions
            unique_input_ind = np.where(np.in1d(motor_ind,hidden_ind)==False)[0]
            motor_act = self.motorRuleData[sensoryIndices,:][:,unique_input_ind]

            ##### Concatenate input activations
            input_activations = np.hstack((input_act,logic_act,sensory_act,motor_act))

            #### **** #### START HERE

#            if thresh!=None:
#                actflow_taskrules = np.multiply(actflow_taskrules,actflow_taskrules>0)
#                actflow_stim = np.multiply(actflow_stim,actflow_stim>0)
#            hiddenlayer_composition = actflow_taskrules + actflow_stim

            respIndex = np.asarray(self.motorRespAll)[sensoryIndices]
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
            
            sensecount += 1

        #actflow = np.mean(actflow,axis=2)
        #actflow = np.divide(actflow,4.0)

        return input_activations_lmid, input_activations_lind, input_activations_rmid, input_activations_rind

    def generateActFlowPredictions_12Rule_PCFC(self,thresh=0,verbose=False):
        """
        Run all predictions for all 64 tasks
        """
        ### Threshold
        #if thresh==None:
        #    pass
        #else:
        #    self.logicRuleData = np.multiply(self.logicRuleData,self.logicRuleData>thresh)
        #    self.sensoryRuleData = np.multiply(self.sensoryRuleData,self.sensoryRuleData>thresh)
        #    self.motorRuleData = np.multiply(self.motorRuleData,self.motorRuleData>thresh)

        target_vertices = self.fc_hidden2motorresp.shape[1]
        actflow = np.zeros((target_vertices,4)) #LMID, LIND, RMID, rIND -- 4 cols in 3rd dim for each sensory rule
        sensecount = 0
        for sensoryRule in self.sensoryRules:
            sensoryIndices = np.where(np.asarray(self.sensoryRuleIndices)==sensoryRule)[0]

            input_ind, rule12_ind, hidden_ind = self.getRoiIndices(sensoryRule,rule12=True) # Can randomly choose logic/motor rules - doesn't maek a difference. need to edit this

            # Run activity flow

            #### Input to hidden regions
            # first identify non-overlapping indices
            unique_input_ind = np.where(np.in1d(input_ind,hidden_ind)==False)[0]
            fc = self.fc_input2hidden[sensoryRule]
            pc_act = np.matmul(self.stimData[sensoryIndices,:][:,unique_input_ind],self.eig_input2hidden[sensoryRule].T)
            actflow_stim = np.matmul(pc_act,fc) 
            
            #### Rule compositions
            ####  (12rule) to hidden regions
            unique_input_ind = np.where(np.in1d(rule12_ind,hidden_ind)==False)[0]
            rule_composition = self.logicRuleData[sensoryIndices,:][:,unique_input_ind] + self.sensoryRuleData[sensoryIndices,:][:,unique_input_ind] + self.motorRuleData[sensoryIndices,:][:,unique_input_ind]
            # first identify non-overlapping indices
            fc = self.fc_12rule2hidden
            pc_act = np.matmul(rule_composition,self.eig_12rule2hidden.T)
            actflow_taskrules = np.matmul(pc_act,fc) 

            #### Compositionality
            #actflow_logicrule = np.multiply(actflow_logicrule>thresh,actflow_logicrule)
            #actflow_sensoryrule = np.multiply(actflow_sensoryrule>thresh,actflow_sensoryrule)
            #actflow_motorrule = np.multiply(actflow_motorrule>thresh,actflow_motorrule)
            ### Compositional representations in hidden layers
            #hiddenlayer_composition = actflow_stim + actflow_logicrule + actflow_sensoryrule + actflow_motorrule

            #actflow_taskrules = actflow_logicrule + actflow_sensoryrule + actflow_motorrule
            #if thresh!=None:
            #    actflow_taskrules = np.multiply(actflow_taskrules,actflow_taskrules>0)
            #    actflow_stim = np.multiply(actflow_stim,actflow_stim>0)
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
            unique_ind = np.where(np.in1d(hidden_ind,self.target_ind)==False)[0]
            fc = self.fc_hidden2motorresp
            pc_act = np.matmul(hiddenlayer_composition[:,unique_ind],self.eig_hidden2motorresp.T)
            actflow_predictions = np.real(np.matmul(pc_act,fc))


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

    def getRoiIndices(self,sensoryRule,n_hiddenregions=10,rule12=False):
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
        rule12_regions = np.loadtxt(ruledir + '12Rule_Regions.csv',delimiter=',')

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
        rule12_ind = []
        for roi in rule12_regions:
            rule12_ind.extend(np.where(glasser2==roi+1)[0])
        hidden_ind = []
        for roi in hiddenregions:
            hidden_ind.extend(np.where(glasser2==roi+1)[0])
        
        if rule12:
            return input_ind, rule12_ind, hidden_ind
        else:
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
        print('Logic rule:', logicRule)
        print('Sensory rule:', sensoryRule)
        print('Motor rule:', motorRule)
        print('**Stimuli**')
        print(stim1, stim2)
        print('Motor response:', motorOutput)

    return motorOutput

def constructTasks(n_stims=1,filename='/projects3/SRActFlow/data/results/GroupfMRI/RSA/EmpiricalSRActFlow_AllTrialKeys1.h5'):
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

def constructTasksForRITLGeneralization(n_stims=1,filename_training='/projects3/SRActFlow/data/results/GroupfMRI/RSA/GroupfMRI14a_EmpiricalSRActFlow_TrainingTasks.csv',
                                        filename_testing='/projects3/SRActFlow/data/results/GroupfMRI/RSA/GroupfMRI14a_EmpiricalSRActFlow_TestTasks.csv'):
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


