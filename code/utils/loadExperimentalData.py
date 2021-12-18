# Taku Ito
# 01/15/2018
# 
# Module to load in experimental files for within-subject analyses

import numpy as np
import pandas as pd

def loadExperimentalData(subj):
    """
    Loads in relevant data about the task/experiment/behavior for a given subject
    Pulls in data from the EDAT file
    """
    edat_dir = '/projects/IndivRITL/data/rawdata/' + subj + '/behavdata/fMRI_Behavioral/'
    
    n_miniblocks = 128
    n_trials = 384 # 128 mini blocks + 3 trials

    # Instantiate dict to hold data
    df = {}
    edat = pd.read_csv(edat_dir + subj + '_fMRI_CPRO.txt', delimiter='\t')

    # Motor behavior
    df['MotorResponses'] = edat['ProbeStim1.RESP'][-n_trials:].values
    # Task rules
    df['LogicRules'] = edat['LogicCue[LogLevel5]'][-n_trials:].values
    df['SensoryRules'] = edat['SemanticCue[LogLevel5]'][-n_trials:].values
    df['MotorRules'] = edat['ResponseCue[LogLevel5]'][-n_trials:].values
    # Task novelty
    df['TaskNovelty'] = edat['TaskType_rec'][-n_trials:].values
    # Task number
    df['TaskID'] = edat['TaskName[LogLevel5]'][-n_trials:].values
    # Task performance
    df['TaskPerformance'] = edat['Feedback[LogLevel6]'][-n_trials:].values
    # Stimulus information
    # Visual info
    df['Stim1_Color'] = edat['stim1_color[LogLevel6]'][-n_trials:].values
    df['Stim1_Ori'] = edat['stim1_orientation[LogLevel6]'][-n_trials:].values
    df['Stim2_Color'] = edat['stim2_color[LogLevel6]'][-n_trials:].values
    df['Stim2_Ori'] = edat['stim2_orientation[LogLevel6]'][-n_trials:].values
    # Auditory info
    df['Stim1_Pitch'] = edat['stim1_pitch[LogLevel6]'][-n_trials:].values
    df['Stim1_Constant'] = edat['stim1_constant[LogLevel6]'][-n_trials:].values
    df['Stim2_Pitch'] = edat['stim2_pitch[LogLevel6]'][-n_trials:].values
    df['Stim2_Constant'] = edat['stim2_constant[LogLevel6]'][-n_trials:].values


    df = pd.DataFrame(df)
    return df

