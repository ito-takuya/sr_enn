# Taku Ito
# 05/10/2019
# RSA analysis for RNN model training with no trial dynamics
import pandas as pd
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import task
import multiprocessing as mp
import h5py
from importlib import reload
task = reload(task)
import time
import matplotlib.pyplot as plt
import seaborn as sns


basedir = '/home/ti61/f_mc1689_1/SRActFlow/'



def rsa(network,show=True, savepdf=False):
    """
    For each input element, inject a single element representing each rule/stimulus
    Observe the representational space
    """
    # ordering of the task rules, 1-12
    rule_order = ['BOTH', 'NOTBOTH', 'EITHER', 'NEITHER', 'RED', 'VERTICAL', 'HIGH', 'CONSTANT', 'LMID', 'LIND', 'RIND', 'RMID']
    # ordering of stim input nodes, 12-28
    stim_order = ['RED1', 'BLUE1', 'VERTICAL1', 'HORIZONTAL1', 'HIGH1', 'LOW1', 'CONSTANT1', 'BEEPING1', 'RED2', 'BLUE2', 'VERTICAL2', 'HORIZONTAL2', 'HIGH2', 'LOW2', 'CONSTANT2', 'BEEPING2']

    # combined rule and input element order label array
    input_order = np.hstack((np.asarray(rule_order), np.asarray(stim_order)))

    # These are the representations we want to compare to fMRI data
    rsa_inputs = ['BOTH', 'NOTBOTH', 'EITHER', 'NEITHER', 
                  'RED', 'VERTICAL', 'HIGH', 'CONSTANT', 
                  'LMID', 'LIND', 'RIND', 'RMID',
                  'RED1 RED2', 'RED1 BLUE2', 'BLUE1 RED2', 'BLUE1 BLUE2',
                  'VERTICAL1 VERTICAL2','VERTICAL1 HORIZONTAL2', 'HORIZONTAL1 VERTICAL2', 'HORIZONTAL1 HORIZONTAL2',
                  'HIGH1 HIGH2', 'HIGH1 LOW2', 'LOW1 HIGH2', 'LOW1 LOW2',
                  'CONSTANT1 CONSTANT2', 'CONSTANT1 BEEPING2', 'BEEPING1 CONSTANT2', 'BEEPING1 BEEPING2']

    rule_ind = np.arange(network.num_rule_inputs) # rules are the first 12 indices of input vector
    stim_ind = np.arange(network.num_rule_inputs, network.num_rule_inputs+network.num_sensory_inputs)
    input_size = len(rule_ind) + len(stim_ind)
    input_matrix = np.zeros((input_size,input_size)) # Activation for each input element separately


    input_col = 0
    for element in rsa_inputs:
        # First evaluate if it's a rule
        if input_col < len(rule_ind):
            # Find where the elements are ordered
            input_element = np.where(input_order==element)[0]
            input_matrix[input_element,input_col] = 1.0
        else:
            # Otherwise, this is a stimulus pairing
            tmp_stims = element.split() # Split by space (stimuli are separated by a space) 
            for stim in tmp_stims:
                input_element = np.where(input_order==stim)[0]
                input_matrix[input_element,input_col] = 1.0

        # Go to next input
        input_col += 1

    input_matrix = torch.from_numpy(input_matrix).float()
    # Now run a forward pass for all activations
    outputs, hidden_layers = network.forward(input_matrix,noise=False)

    n_layers = network.n_hidden_layers
    rsm = []
    for layer in range(n_layers):
        tmp = np.corrcoef(hidden_layers[:,:,layer])
        np.fill_diagonal(tmp,0)
        rsm.append(tmp)

    if show:
        plt.figure(figsize=(12,12))
        plt.title('Representational similarity matrix\nof hidden units',fontsize=28)
        ax = sns.heatmap(rsm,square=True,center=0,cmap='bwr', cbar=True,cbar_kws={'fraction':0.046})
        plt.xlabel('Rule + Stimulus representations',fontsize=20)
        plt.ylabel('Rule + Stimulus representations',fontsize=20)
        plt.xticks(np.arange(0.5, len(rsa_inputs)+1), rsa_inputs, rotation=90,fontsize=14)
        plt.yticks(np.arange(0.5, len(rsa_inputs)+1), rsa_inputs, rotation=0, fontsize=14)
        plt.tight_layout()
        ax.invert_yaxis()
        if savepdf:
            plt.savefig('ANN_RSM.pdf')

    return hidden_layers, rsm
    

