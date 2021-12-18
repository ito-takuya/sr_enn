# Taku Ito
# 6/9/21
# 
# Module to load in experimental files for within-subject analyses and convert to BIDS-compatible events.tsv file

import numpy as np
import pandas as pd
import argparse

# good subjs
subjNums = ['013','014','016','017','018','021','023','024','026','027','028','030','031','032','033','034','035','037','038','039','040','041','042','043','045','046','047','048','049','050','053','055','056','057','058','062','063','066','067','068','069','070','072','074','075','076','077','081','085','086','087','088','090','092','093','094','095','097','098','099','101','102','103','104','105','106','108','109','110','111','112','114','115','117','119','120','121','122','123','124','125','126','127','128','129','130','131','132','134','135','136','137','138','139','140','141']

# bids directory on linux server
bids_dir = '/projects/IndivRITL/bids/'

# parser
parser = argparse.ArgumentParser('./main.py', description='Run RSM analysis for each parcel using vertex-wise activations')
parser.add_argument('--subj', type=str, default="all", help='Subject ID (Default: all')

def run(args):
    args 
    subj = args.subj


    if subj == 'all':
        for sub in subjNums:
            print('Converting behavioral eprime data to bids format, subject', sub)
            convert2bids_forsubj(sub)
    else:
        print('Converting behavioral eprime data to bids format, subject', subj)
        convert2bids_forsubj(subj)

def convert2bids_forsubj(subj):
    #### Parameters
    ### Mini block design - 3 trials per miniblock
    n_tr_per_run = 581
    n_trials_per_block = 3
    n_task_runs = 8
    n_blocks_per_run = 16
    n_trs_per_block = 36
    n_trs_per_encoding = 5
    n_trs_per_trial = 3
    tr_value = 785 # ms
    iti_length = 2 # trs
    # total
    n_miniblocks = 128
    n_trials = 384 # 128 mini blocks + 3 trials

    # Start
    sub = subj

    edat_dir = '/projects/IndivRITL/data/rawdata/' + sub + '/behavdata/fMRI_Behavioral/'
    # Instantiate dict to hold data
    df = {}
    edat = pd.read_csv(edat_dir + sub + '_fMRI_CPRO.txt', delimiter='\t')

    #####
    # Trial timing info
    df['firstjitter'] = edat['blockITI'][-n_trials:].values
    df['secondjitter'] = edat['InterBlockInterval[LogLevel5]'][-n_trials:].values
    # probe delay
    df['probedelay'] = edat['probedelay[LogLevel6]'][-n_trials:].values
    # Motor behavior - remap 'b', 'y', 'g', 'r', to 'lmid', 'lind', 'rind', 'rmid' fingers respectively
    tmpresp = edat['ProbeStim1.RESP'][-n_trials:].values
    ind = tmpresp=='b'
    tmpresp[ind] = 'left_middle_finger'
    ind = tmpresp=='y'
    tmpresp[ind] = 'left_index_finger'
    ind = tmpresp=='g'
    tmpresp[ind] = 'right_index_finger'
    ind = tmpresp=='r'
    tmpresp[ind] = 'right_middle_finger'
    df['MotorResponses'] = tmpresp 
    df['RT'] = edat['ProbeStim1.RT'][-n_trials:].values
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
    #
    df = pd.DataFrame(df)

    i = 0 # trial counter, across all runs
    for run in range(n_task_runs):
        onset = 0 #  onsets for each run 

        # Now get TR onsets and duration_in_trs for each category
        event_df = {}
        event_df['onset_in_tr'] = []
        event_df['duration_in_tr'] = []
        event_df['task_event'] = [] 
        event_df['rt_in_ms'] = []
        event_df['task_novelty'] = []
        event_df['logic_rule'] = []
        event_df['sensory_rule'] = []
        event_df['motor_rule'] = []
        event_df['performance'] = []
        event_df['task_id'] = []
        event_df['motor_response'] = []
        # 2 stimuli presented consecutively per trial
        event_df['stimulus1_color'] = []
        event_df['stimulus1_orientation'] = []
        event_df['stimulus1_pitch'] = []
        event_df['stimulus1_continuity'] = []
        event_df['stimulus2_color'] = []
        event_df['stimulus2_orientation'] = []
        event_df['stimulus2_pitch'] = []
        event_df['stimulus2_continuity'] = []

        #### First 5 TRs are delay
        event_df['onset_in_tr'].append(0)
        event_df['duration_in_tr'].append(5)
        event_df['task_event'].append('delay')
        event_df['rt_in_ms'].append('na')
        event_df['task_novelty'].append('na')
        event_df['logic_rule'].append('na')
        event_df['sensory_rule'].append('na')
        event_df['motor_rule'].append('na')
        event_df['performance'].append('na')
        event_df['task_id'].append('na')
        event_df['motor_response'].append('na')
        event_df['stimulus1_color'].append('na')
        event_df['stimulus1_orientation'].append('na')
        event_df['stimulus1_pitch'].append('na')
        event_df['stimulus1_continuity'].append('na')
        event_df['stimulus2_color'].append('na')
        event_df['stimulus2_orientation'].append('na')
        event_df['stimulus2_pitch'].append('na')
        event_df['stimulus2_continuity'].append('na')
        # Increase onset
        onset += 5

        #### Iterate through each block
        for block in range(n_blocks_per_run):
            #### ENCODING PERIOD
            event_df['onset_in_tr'].append(onset)
            event_df['duration_in_tr'].append(n_trs_per_encoding)
            event_df['task_event'].append('encoding')
            event_df['rt_in_ms'].append('na')
            event_df['task_novelty'].append(df.TaskNovelty.values[i])
            event_df['logic_rule'].append(df.LogicRules.values[i])
            event_df['sensory_rule'].append(df.SensoryRules.values[i])
            event_df['motor_rule'].append(df.MotorRules.values[i])
            event_df['performance'].append('na')
            event_df['task_id'].append(df.TaskID.values[i])
            event_df['motor_response'].append('na')
            event_df['stimulus1_color'].append('na')
            event_df['stimulus1_orientation'].append('na')
            event_df['stimulus1_pitch'].append('na')
            event_df['stimulus1_continuity'].append('na')
            event_df['stimulus2_color'].append('na')
            event_df['stimulus2_orientation'].append('na')
            event_df['stimulus2_pitch'].append('na')
            event_df['stimulus2_continuity'].append('na')
            onset += n_trs_per_encoding
            #
            #
            #### JiTTERED DELAY
            event_df['onset_in_tr'].append(onset)
            event_df['duration_in_tr'].append(df.firstjitter.values[i]/tr_value)
            event_df['task_event'].append('delay')
            event_df['rt_in_ms'].append('na')
            event_df['task_novelty'].append('na')
            event_df['logic_rule'].append('na')
            event_df['sensory_rule'].append('na')
            event_df['motor_rule'].append('na')
            event_df['performance'].append('na')
            event_df['task_id'].append('na')
            event_df['motor_response'].append('na')
            event_df['stimulus1_color'].append('na')
            event_df['stimulus1_orientation'].append('na')
            event_df['stimulus1_pitch'].append('na')
            event_df['stimulus1_continuity'].append('na')
            event_df['stimulus2_color'].append('na')
            event_df['stimulus2_orientation'].append('na')
            event_df['stimulus2_pitch'].append('na')
            event_df['stimulus2_continuity'].append('na')
            onset += df.firstjitter.values[i]/tr_value
            #
            #
            #### 3 Trials per block
            # Iterate through each trial and add to trial counter
            for trial in range(n_trials_per_block):
                event_df['onset_in_tr'].append(onset)
                event_df['duration_in_tr'].append(n_trs_per_trial)
                event_df['task_event'].append('trial')
                event_df['rt_in_ms'].append(df.RT.values[i])
                event_df['task_novelty'].append(df.TaskNovelty.values[i])
                event_df['logic_rule'].append(df.LogicRules.values[i])
                event_df['sensory_rule'].append(df.SensoryRules.values[i])
                event_df['motor_rule'].append(df.MotorRules.values[i])
                event_df['performance'].append(df.TaskPerformance.values[i])
                event_df['task_id'].append(df.TaskID.values[i])
                event_df['motor_response'].append(df.MotorResponses.values[i])
                event_df['stimulus1_color'].append(df.Stim1_Color.values[i])
                event_df['stimulus1_orientation'].append(df.Stim1_Ori.values[i])
                event_df['stimulus1_pitch'].append(df.Stim1_Pitch.values[i])
                event_df['stimulus1_continuity'].append(df.Stim1_Constant.values[i])
                event_df['stimulus2_color'].append(df.Stim2_Color.values[i])
                event_df['stimulus2_orientation'].append(df.Stim2_Ori.values[i])
                event_df['stimulus2_pitch'].append(df.Stim2_Pitch.values[i])
                event_df['stimulus2_continuity'].append(df.Stim2_Constant.values[i])
                onset += n_trs_per_trial
                #
                # Include inter-trial or inter-block interval
                # If last trial use interblock interval (variable jitter)
                if trial==2:
                    event_df['onset_in_tr'].append(onset)
                    event_df['duration_in_tr'].append(df.secondjitter.values[i]/tr_value)
                    event_df['task_event'].append('delay')
                    event_df['rt_in_ms'].append('na')
                    event_df['task_novelty'].append('na')
                    event_df['logic_rule'].append('na')
                    event_df['sensory_rule'].append('na')
                    event_df['motor_rule'].append('na')
                    event_df['performance'].append('na')
                    event_df['task_id'].append('na')
                    event_df['motor_response'].append('na')
                    event_df['stimulus1_color'].append('na')
                    event_df['stimulus1_orientation'].append('na')
                    event_df['stimulus1_pitch'].append('na')
                    event_df['stimulus1_continuity'].append('na')
                    event_df['stimulus2_color'].append('na')
                    event_df['stimulus2_orientation'].append('na')
                    event_df['stimulus2_pitch'].append('na')
                    event_df['stimulus2_continuity'].append('na')
                    onset += df.secondjitter.values[i]/tr_value
                else:
                    # Use inter-trial interval (fixed)
                    event_df['onset_in_tr'].append(onset)
                    event_df['duration_in_tr'].append(iti_length)
                    event_df['task_event'].append('ITI')
                    event_df['rt_in_ms'].append('na')
                    event_df['task_novelty'].append('na')
                    event_df['logic_rule'].append('na')
                    event_df['sensory_rule'].append('na')
                    event_df['motor_rule'].append('na')
                    event_df['performance'].append('na')
                    event_df['task_id'].append('na')
                    event_df['motor_response'].append('na')
                    event_df['stimulus1_color'].append('na')
                    event_df['stimulus1_orientation'].append('na')
                    event_df['stimulus1_pitch'].append('na')
                    event_df['stimulus1_continuity'].append('na')
                    event_df['stimulus2_color'].append('na')
                    event_df['stimulus2_orientation'].append('na')
                    event_df['stimulus2_pitch'].append('na')
                    event_df['stimulus2_continuity'].append('na')
                    onset += iti_length

                # trial counter
                i += 1
                ##


        event_df = pd.DataFrame(event_df)
        event_df.to_csv(bids_dir + 'sub-' + sub + '/ses-01/func/sub-' + sub + '_ses-01_task-cpro_run-0' + str(run+1) +'_events.tsv',sep='\t')


if __name__ == '__main__':
    args = parser.parse_args()
    run(args)
