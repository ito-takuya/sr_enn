# Module to construct task features (e.g., inputs/outputs of tasks) that are independent of the model
# Is used as a 'helper' module for model.py
# Inputs and outputs of CPRO taks
import numpy as np
import torch
import pandas as pd

def createSensoryInputs(nStims=2):
    stimdata = {}
    # Stim 1 empty columns
    stimdata['Color1'] = []
    stimdata['Orientation1'] = []
    stimdata['Pitch1'] = []
    stimdata['Constant1'] = []
    # Stim 2 empty columns
    stimdata['Color2'] = []
    stimdata['Orientation2'] = []
    stimdata['Pitch2'] = []
    stimdata['Constant2'] = []
    # Code for RNN training
    stimdata['Code'] = []

    # Property index tells us which columns ID the property in question
    color = {0:'red',
             1:'blue'}
    orientation = {2:'vertical',
                   3:'horizontal'}
    pitch = {4:'high',
             5:'low'}
    constant = {6:'constant',
                7:'beeping'}
    
    for col1 in color:
        for col2 in color:
            for ori1 in orientation:
                for ori2 in orientation:
                    for pit1 in pitch:
                        for pit2 in pitch:
                            for con1 in constant:
                                for con2 in constant:
                                    code = np.zeros((8*nStims,))
                                    # Stim 1
                                    code[col1] = 1
                                    stimdata['Color1'].append(color[col1])
                                    code[ori1] = 1
                                    stimdata['Orientation1'].append(orientation[ori1])
                                    code[pit1] = 1
                                    stimdata['Pitch1'].append(pitch[pit1])
                                    code[con1] = 1
                                    stimdata['Constant1'].append(constant[con1])
                                    # Stim 2 -- need to add 8, since this is the second stimuli
                                    code[col2+8] = 1
                                    stimdata['Color2'].append(color[col2])
                                    code[ori2+8] = 1
                                    stimdata['Orientation2'].append(orientation[ori2])
                                    code[pit2+8] = 1
                                    stimdata['Pitch2'].append(pitch[pit2])
                                    code[con2+8] = 1
                                    stimdata['Constant2'].append(constant[con2])
                                    
                                    # Code
                                    stimdata['Code'].append(code)
                    
    return pd.DataFrame(stimdata) 

def createRulePermutations():
    # May need to change this - both and not both are negations of each other, as are either and neither
    logicRules = {0: 'both',
                  1: 'notboth',
                  2: 'either',
                  3: 'neither'}
    sensoryRules = {4: 'red',
                    5: 'vertical',
                    6: 'high',
                    7: 'constant'}
    motorRules = {8: 'l_mid',
                  9: 'l_ind',
                  10: 'r_ind',
                  11: 'r_mid'}
    
    
    taskrules = {}
    taskrules['Logic'] = []
    taskrules['Sensory'] = []
    taskrules['Motor'] = []
    # Create another field for the sensory category (to select stimuli from)
    taskrules['SensoryCategory'] = []
    # For RNN training
    taskrules['Code'] = []
    
    for lo in logicRules:
        for se in sensoryRules:
            for mo in motorRules:
                code = np.zeros((12,))
                # Logic rule
                taskrules['Logic'].append(logicRules[lo])
                code[lo] = 1
                
                # Sensory rule
                taskrules['Sensory'].append(sensoryRules[se])
                code[se] = 1
                # Define sensory category
                if sensoryRules[se]=='red': category = 'Color'
                if sensoryRules[se]=='vertical': category = 'Orientation'
                if sensoryRules[se]=='high': category = 'Pitch'
                if sensoryRules[se]=='constant': category = 'Constant'
                taskrules['SensoryCategory'].append(category)
                
                # Motor rule
                taskrules['Motor'].append(motorRules[mo])
                code[mo] = 1
                
                taskrules['Code'].append(code)
                
    return pd.DataFrame(taskrules)


motorCode = {0:'l_mid',
             1:'l_ind',
             2:'r_ind',
             3:'r_mid'}

def solveInputs(task_rules, stimuli, printTask=False):
    """
    Solves CPRO task given a set of inputs and a task rule
    """
    logicRule = task_rules.Logic
    sensoryRule = task_rules.Sensory
    motorRule = task_rules.Motor

    sensoryCategory = task_rules.SensoryCategory

    # Isolate the property for each stimulus relevant to the sensory rule
    stim1 = stimuli[sensoryCategory + '1']
    stim2 = stimuli[sensoryCategory + '2']

    # Run through logic rule gates
    if logicRule == 'both':
        if stim1==sensoryRule and stim2==sensoryRule:
            gate = True
        else:
            gate = False

    if logicRule == 'notboth':
        if stim1!=sensoryRule or stim2!=sensoryRule:
            gate = True
        else:
            gate = False

    if logicRule == 'either':
        if stim1==sensoryRule or stim2==sensoryRule:
            gate = True
        else:
            gate = False

    if logicRule == 'neither':
        if stim1!=sensoryRule and stim2!=sensoryRule:
            gate = True
        else:
            gate = False


    ## Print task first
    if printTask:
        print('Logic rule:', logicRule)
        print('Sensory rule:', sensoryRule)
        print('Motor rule:', motorRule)
        print('**Stimuli**')
        print(stim1, stim2)

    # Apply logic gating to motor rules
    if motorRule=='l_mid':
        if gate==True:
            motorOutput = 'l_mid'
        else:
            motorOutput = 'l_ind'

    if motorRule=='l_ind':
        if gate==True:
            motorOutput = 'l_ind'
        else:
            motorOutput = 'l_mid'

    if motorRule=='r_mid':
        if gate==True:
            motorOutput = 'r_mid'
        else:
            motorOutput = 'r_ind'

    if motorRule=='r_ind':
        if gate==True:
            motorOutput = 'r_ind'
        else:
            motorOutput = 'r_mid'

    outputcode = np.zeros((4,))
    if motorOutput=='l_mid': outputcode[0] = 1
    if motorOutput=='l_ind': outputcode[1] = 1
    if motorOutput=='r_mid': outputcode[2] = 1
    if motorOutput=='r_ind': outputcode[3] = 1

    return motorOutput, outputcode


def createTrainTestTaskRules(taskRuleSet,nTrainSet=32,nTestSet=32):
    """
    Ensure that when we split the task rules, that each set has equal proportions of each task rule
    For example, if there are 32 training tasks, then we should have 8 examples of each rule
    """
    nRulesPerTrainSet = nTrainSet/4.0
    nRulesPerTestSet = nTestSet/4.0
    if nRulesPerTrainSet%4!=0:
        print('WARNING: Number of rules per train/test set is not divisible by 4')
        #raise Exception('ERROR: Number of rules per train/test set needs to be divisible by 4!')

    df_test = pd.DataFrame()
    df_train = pd.DataFrame()
    df_test = []
    df_train = []
    # Make sure all columns exist 
    #df_train = df_train.append(taskRuleSet.iloc[0])

    # Iterate through tasks in a random manner
    ind = np.arange(len(taskRuleSet))
    np.random.shuffle(ind)
    taskcount = 0
    for i in ind:
        # Identify the rules in this task set
        #logic = taskRuleSet.Logic[i]
        #sensory = taskRuleSet.Sensory[i]
        #motor = taskRuleSet.Motor[i]

        ## Count number of logic rules for this task set
        #nLogic = np.sum(df_train.Logic==logic)
        #nSensory = np.sum(df_train.Sensory==sensory)
        #nMotor = np.sum(df_train.Motor==motor)
        #if nLogic<nRulesPerTrainSet and nSensory<nRulesPerTrainSet and nMotor<nRulesPerTrainSet:
        #    df_train = df_train.append(taskRuleSet.iloc[i])
        #else:
        #    df_test = df_test.append(taskRuleSet.iloc[i])
        if taskcount<nTrainSet:
            df_train.append(taskRuleSet.iloc[i])
        else:
            df_test.append(taskRuleSet.iloc[i])

        taskcount += 1

    df_test = pd.DataFrame(df_test)
    df_train = pd.DataFrame(df_train)

    return df_train, df_test
