# Taku Ito
# 02/22/2017

# Utility package for functions useful for StroopActFlow

import numpy as np

def loadBehavData(subj):
    """
    Loads in behavioral data for a given subject as a dict
    """

    behavdir = '/projects3/StroopActFlow/data/results/behavdata/'

    data = {}
    data['accuracy'] = np.loadtxt(behavdir+subj+'_accuracy.csv',delimiter=',')
    data['colorStim'] = np.loadtxt(behavdir+subj+'_colorStim.csv',delimiter=',',dtype=str)
    data['condition'] = np.loadtxt(behavdir+subj+'_condition.csv',delimiter=',',dtype=str)
    data['block_rule'] = np.loadtxt(behavdir+subj+'_miniblockRuleEncodings.csv',delimiter=',',dtype=str)
    data['rt'] = np.loadtxt(behavdir+subj+'_RT.csv',delimiter=',')
    data['response'] = np.loadtxt(behavdir+subj+'_subjResponse.csv',delimiter=',',dtype=str)
    data['taskRule'] = np.loadtxt(behavdir+subj+'_taskRule.csv',delimiter=',',dtype=str)
    data['wordStim'] = np.loadtxt(behavdir+subj+'_wordStim.csv',delimiter=',',dtype=str)

    return data

