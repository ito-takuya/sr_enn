# Taku Ito
# 8/3/2020
# General function modules for SRActFlow
# For group-level/cross-subject analyses

import numpy as np
import multiprocessing as mp
import scipy.stats as stats
import nibabel as nib
import sklearn
import h5py
from scipy import signal
import sys
import os
os.environ['OMP_NUM_THREADS'] = str(1)
os.sys.path.append('glmScripts/')
sys.path.append('utils/')
import loadExperimentalData as led
import tools
import pathlib
import time
import calculateFC as fc
import pandas as pd
import SRModels 
import argparse
projectdir = '/home/ti61/f_mc1689_1/SRActFlow/'

parser = argparse.ArgumentParser('.main.py', description='Run ENNs')
parser.add_argument('--model', type=str, default='full', help='Type of models to run [full, norelu, nohidden, contextlesion, shufflefc, randomhidden,fullnetwork]')
parser.add_argument('--nproc', type=int, default=10, help='number of parallel cpus')
parser.add_argument('--ncomponents', type=int, default=500, help='number of principal components for FC estimation')
parser.add_argument('--nhidden', type=int, default=10, help='number of hidden units')
parser.add_argument('--computeFC', action='store_true',help='recompute FC (if not, load from memory)')
parser.add_argument('--iteration', type=int, default=0, help='random iteration (only used/relevant for "randomhidden" model)')
parser.add_argument('--scratchfcdir', type=str, default='', help='scratch dir to save out random fc matrices (only used/relevant for "randomhidden" model)')

def run(args):
    args
    model = args.model
    nproc = args.nproc
    ncomponents = args.ncomponents
    nhidden = args.nhidden
    computeFC = args.computeFC
    iteration = args.iteration
    scratchfcdir = args.scratchfcdir
    projectdir = '/home/ti61/f_mc1689_1/SRActFlow/'


    if model=='full':
        runFullSRModel(projectdir=projectdir,outdir=projectdir+'data/results/MAIN/srModelPredictionAccuracies/SRActFlowFull/',
                           thresh=0,ruletype='12',nhidden=nhidden,computeFC=computeFC,ncomponents=ncomponents,nproc=nproc)
    if model=='norelu':
        runNoReLUSRModel(projectdir=projectdir,outdir=projectdir+'data/results/MAIN/srModelPredictionAccuracies/SRActFlowNoReLU/',
                             nhidden=nhidden,computeFC=computeFC,ncomponents=ncomponents,nproc=nproc)

    if model=='nohidden':
        runNoHiddenSRModel(projectdir=projectdir,outdir=projectdir+'data/results/MAIN/srModelPredictionAccuracies/SRActFlowNoHidden/',
                             computeFC=computeFC,ncomponents=ncomponents,nproc=nproc)

    if model=='contextlesion':
        runSRModelContextLesion(projectdir=projectdir,outdir=projectdir+'data/results/MAIN/srModelPredictionAccuracies/SRActFlowContextLesion/',
                                    thresh=0,nhidden=nhidden,computeFC=computeFC,ncomponents=ncomponents,nproc=nproc)

    if model=='shufflefc':
        runSRModelShuffleFC(projectdir=projectdir,outdir=projectdir+'data/results/MAIN/srModelPredictionAccuracies/SRActFlowShuffleFC/',
                                    thresh=0,nhidden=nhidden,computeFC=computeFC,ncomponents=ncomponents,nproc=nproc,
                                    seed=np.random.randint(0,100000))

    if model=='randomhidden':
        runRandomizedHiddenLayerModel_v2(projectdir=projectdir,outdir=projectdir+'data/results/MAIN/srModelPredictionAccuracies/SRActFlowRandomizedHidden/',
                   thresh=0,nhidden=nhidden,computeFC=computeFC,ncomponents=ncomponents,nproc=nproc,scratchfcdir=scratchfcdir,iteration=iteration)

    #### reviewer request
    if model=='hiddenrsm':
        # generally speaking, you never need to alter the parameters of this function
        extractHiddenRSM(projectdir=projectdir,outdir=projectdir+'data/results/MAIN/srModelPredictionAccuracies/hiddenRSMs/',
                           nhidden=nhidden,computeFC=computeFC,ncomponents=ncomponents,nproc=nproc)

    ### Reviewer request -- run full model but decoding on all smn regions with decodable representations
    if model=='fullnetwork':
        runFullSRModel(projectdir=projectdir,outdir=projectdir+'data/results/MAIN/srModelPredictionAccuracies/SRActFlowFullSMNNetwork/',
                           thresh=0,ruletype='12',nhidden=nhidden,computeFC=computeFC,vertexmasks=False,featsel=True,ncomponents=ncomponents,nproc=nproc)

        



def runFullSRModel(projectdir=projectdir,outdir=projectdir+'data/results/MAIN/srModelPredictionAccuracies/SRActFlowFull/',
                   thresh=0,ruletype='12',nhidden=10,computeFC=False,vertexmasks=True,featsel=False,ncomponents=500,nproc=10):
    """
    Full SR model 
        vertexmasks: predict on digit representations only (identified via t-test). If False, decode on all smn regions that contained decodable information
    """
    #### Don't change this parameter -- override
    thresh = 0
    ####

    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True) # Make sure directory exists

    Model = SRModels.Model(projectdir=projectdir,ruletype=ruletype,n_hiddenregions=nhidden,randomize=False)

    #### Only compute FC if it has not been computed before -- computationally intensive process
    if computeFC:
        Model.computeGroupFC(n_components=ncomponents,nproc=nproc)

    #### Load real activations onto model
    Model.loadRealMotorResponseActivations(vertexmasks=vertexmasks)

    #### Load group FC weights as model weights
    Model.loadModelFC()

    #### Run SRActFlow simulations on entire group
    actflow_rh, actflow_lh = Model.simulateGroupActFlow(thresh=thresh,nproc=nproc,vertexmasks=vertexmasks)

    #### Evaluate how well actflow performs  relative to actual activations
    nbootstraps = 1000
    print('Run decoding on RH actflow predictions...')
    rh_filename = outdir + 'RH_decoding.txt'
    Model.actflowDecoding(actflow_rh, Model.data_task_rh, rh_filename,
                          nbootstraps=nbootstraps, nproc=nproc, featsel=featsel, null=False, verbose=True)

    print('Run null (permutation) decoding on RH actflow predictions...')
    rh_filename_null = outdir + 'RH_null_decoding.txt'
    Model.actflowDecoding(actflow_rh, Model.data_task_rh, rh_filename_null,
                          nbootstraps=nbootstraps, nproc=nproc, featsel=featsel, null=True, verbose=True)

    print('Run decoding on LH actflow predictions...')
    lh_filename = outdir + 'LH_decoding.txt'
    Model.actflowDecoding(actflow_lh, Model.data_task_lh, lh_filename,
                          nbootstraps=nbootstraps, nproc=nproc, featsel=featsel, null=False, verbose=True)

    print('Run null (permutation) decoding on LH actflow predictions...')
    lh_filename_null = outdir + 'LH_null_decoding.txt'
    Model.actflowDecoding(actflow_lh, Model.data_task_lh, lh_filename_null,
                          nbootstraps=nbootstraps, nproc=nproc, featsel=featsel, null=True, verbose=True)

def runNoReLUSRModel(projectdir=projectdir,outdir=projectdir+'data/results/MAIN/srModelPredictionAccuracies/SRActFlowNoReLU/',
                     nhidden=10,computeFC=False,ncomponents=500,nproc=10):
    """
    No ReLU model - remove nonlinearity in hidden area (see thresh=None)
    """
    #### Don't change this parameter -- override
    nhidden = 10
    thresh = None # Don't include threshold 
    ####

    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True) # Make sure directory exists

    Model = SRModels.Model(projectdir=projectdir,n_hiddenregions=nhidden,randomize=False)

    #### Only compute FC if it has not been computed before -- computationally intensive process
    if computeFC:
        Model.computeGroupFC(n_components=ncomponents,nproc=nproc)

    #### Load real activations onto model
    Model.loadRealMotorResponseActivations()

    #### Load group FC weights as model weights
    Model.loadModelFC()

    #### Run SRActFlow simulations on entire group
    actflow_rh, actflow_lh = Model.simulateGroupActFlow(thresh=thresh,nproc=nproc)

    #### Evaluate how well actflow performs  relative to actual activations
    nbootstraps = 1000
    print('Run decoding on RH actflow predictions...')
    rh_filename = outdir + 'RH_decoding.txt'
    Model.actflowDecoding(actflow_rh, Model.data_task_rh, rh_filename,
                          nbootstraps=nbootstraps, nproc=nproc, null=False, verbose=True)

    print('Run null (permutation) decoding on RH actflow predictions...')
    rh_filename_null = outdir + 'RH_null_decoding.txt'
    Model.actflowDecoding(actflow_rh, Model.data_task_rh, rh_filename_null,
                          nbootstraps=nbootstraps, nproc=nproc, null=True, verbose=True)

    print('Run decoding on LH actflow predictions...')
    lh_filename = outdir + 'LH_decoding.txt'
    Model.actflowDecoding(actflow_lh, Model.data_task_lh, lh_filename,
                          nbootstraps=nbootstraps, nproc=nproc, null=False, verbose=True)

    print('Run null (permutation) decoding on LH actflow predictions...')
    lh_filename_null = outdir + 'LH_null_decoding.txt'
    Model.actflowDecoding(actflow_lh, Model.data_task_lh, lh_filename_null,
                          nbootstraps=nbootstraps, nproc=nproc, null=True, verbose=True)

def runNoHiddenSRModel(projectdir=projectdir,outdir=projectdir+'data/results/MAIN/srModelPredictionAccuracies/SRActFlowNoHidden/',
                     computeFC=False,ncomponents=500,nproc=10):
    """
    No Hidden Layer model - remove hidden layer entirely 
    """
    #### Don't change this parameter
    nhidden = None
    thresh = 0 # Include a threshold on output layer
    ####

    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True) # Make sure directory exists

    Model = SRModels.Model(projectdir=projectdir,n_hiddenregions=nhidden,randomize=False)

    #### Only compute FC if it has not been computed before -- computationally intensive process
    if computeFC:
        Model.computeGroupFC(n_components=ncomponents,nproc=nproc)

    #### Load real activations onto model
    Model.loadRealMotorResponseActivations()

    #### Load group FC weights as model weights
    Model.loadModelFC()

    #### Run SRActFlow simulations on entire group
    actflow_rh, actflow_lh = Model.simulateGroupActFlow(thresh=thresh,nproc=nproc)

    #### Evaluate how well actflow performs  relative to actual activations
    nbootstraps = 1000
    print('Run decoding on RH actflow predictions...')
    rh_filename = outdir + 'RH_decoding.txt'
    Model.actflowDecoding(actflow_rh, Model.data_task_rh, rh_filename,
                          nbootstraps=nbootstraps, nproc=nproc, null=False, verbose=True)

    print('Run null (permutation) decoding on RH actflow predictions...')
    rh_filename_null = outdir + 'RH_null_decoding.txt'
    Model.actflowDecoding(actflow_rh, Model.data_task_rh, rh_filename_null,
                          nbootstraps=nbootstraps, nproc=nproc, null=True, verbose=True)

    print('Run decoding on LH actflow predictions...')
    lh_filename = outdir + 'LH_decoding.txt'
    Model.actflowDecoding(actflow_lh, Model.data_task_lh, lh_filename,
                          nbootstraps=nbootstraps, nproc=nproc, null=False, verbose=True)

    print('Run null (permutation) decoding on LH actflow predictions...')
    lh_filename_null = outdir + 'LH_null_decoding.txt'
    Model.actflowDecoding(actflow_lh, Model.data_task_lh, lh_filename_null,
                          nbootstraps=nbootstraps, nproc=nproc, null=True, verbose=True)

def runSRModelContextLesion(projectdir=projectdir,outdir=projectdir+'data/results/MAIN/srModelPredictionAccuracies/SRActFlowContextLesion/',
                            thresh=0,nhidden=10,computeFC=False,ncomponents=500,nproc=10):
    """
    SR Model with Context Lesion 
    After loading in FC weights, zero out the context weights
    """
    #### Don't change this parameter -- override
    thresh = 0
    ####

    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True) # Make sure directory exists

    Model = SRModels.Model(projectdir=projectdir,n_hiddenregions=nhidden,randomize=False)

    #### Only compute FC if it has not been computed before -- computationally intensive process
    if computeFC:
        Model.computeGroupFC(n_components=ncomponents,nproc=nproc)

    #### Load real activations onto model
    Model.loadRealMotorResponseActivations()

    #### Load group FC weights as model weights
    Model.loadModelFC()
    #### CONTEXT LESION - SET RULE WEIGHTS TO 0
    print("Running task rule lesion simulation -- setting connections from rule layer to 0!")
    Model.fc_12rule2hidden = np.zeros(Model.fc_12rule2hidden.shape)

    #### Run SRActFlow simulations on entire group
    actflow_rh, actflow_lh = Model.simulateGroupActFlow(thresh=thresh,nproc=nproc)

    #### Evaluate how well actflow performs  relative to actual activations
    nbootstraps = 1000
    print('Run decoding on RH actflow predictions...')
    rh_filename = outdir + 'RH_decoding.txt'
    Model.actflowDecoding(actflow_rh, Model.data_task_rh, rh_filename,
                          nbootstraps=nbootstraps, nproc=nproc, null=False, verbose=True)

    print('Run null (permutation) decoding on RH actflow predictions...')
    rh_filename_null = outdir + 'RH_null_decoding.txt'
    Model.actflowDecoding(actflow_rh, Model.data_task_rh, rh_filename_null,
                          nbootstraps=nbootstraps, nproc=nproc, null=True, verbose=True)

    print('Run decoding on LH actflow predictions...')
    lh_filename = outdir + 'LH_decoding.txt'
    Model.actflowDecoding(actflow_lh, Model.data_task_lh, lh_filename,
                          nbootstraps=nbootstraps, nproc=nproc, null=False, verbose=True)

    print('Run null (permutation) decoding on LH actflow predictions...')
    lh_filename_null = outdir + 'LH_null_decoding.txt'
    Model.actflowDecoding(actflow_lh, Model.data_task_lh, lh_filename_null,
                          nbootstraps=nbootstraps, nproc=nproc, null=True, verbose=True)

def runSRModelShuffleFC(projectdir=projectdir,outdir=projectdir+'data/results/MAIN/srModelPredictionAccuracies/SRActFlowShuffleFC/',
                            thresh=0,nhidden=10,computeFC=False,ncomponents=500,nproc=10,seed=np.random.randint(0,100000)):
    """
    SR Model with FC Shuffled - this function runs a single iteration of a permuted FC channel
    To obtain the 'full' permutation of shuffled FC, use a batch function that calls this function
    After loading in FC weights, zero out the context weights
    """
    #### Don't change this parameter -- override
    thresh = 0
    ####

    print('Random seed:', seed)

    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True) # Make sure directory exists

    Model = SRModels.Model(projectdir=projectdir,n_hiddenregions=nhidden,randomize=False)

    #### Only compute FC if it has not been computed before -- computationally intensive process
    if computeFC:
        Model.computeGroupFC(n_components=ncomponents,nproc=nproc)

    #### Load real activations onto model
    Model.loadRealMotorResponseActivations()

    #### Load group FC weights as model weights
    Model.loadModelFC()
    #### FC Shuffle - Randomly shuffle weights
    print("Shuffling FC connections from the input layer...")
    for stim in Model.fc_input2hidden:
        input_ind = np.arange(Model.fc_input2hidden[stim].shape[0]) 
        np.random.shuffle(input_ind)
        Model.fc_input2hidden[stim] = np.squeeze(Model.fc_input2hidden[stim][input_ind,:])
    # Shuffle connections for context layer
    input_ind = np.arange(Model.fc_12rule2hidden.shape[0]) 
    np.random.shuffle(input_ind)
    Model.fc_12rule2hidden= np.squeeze(Model.fc_12rule2hidden[input_ind,:])


    #### Run SRActFlow simulations on entire group
    actflow_rh, actflow_lh = Model.simulateGroupActFlow(thresh=thresh,nproc=nproc)

    #### Evaluate how well actflow performs  relative to actual activations
    nbootstraps = 1 # Only one bootstrap since permutation will come in the 'outer loop'
    print('Run decoding on RH actflow predictions...')
    rh_filename = outdir + 'RH_decoding.txt'
    Model.actflowDecoding(actflow_rh, Model.data_task_rh, rh_filename,
                          nbootstraps=nbootstraps, nproc=nproc, null=False, verbose=True)

    print('Run null (permutation) decoding on RH actflow predictions...')
    rh_filename_null = outdir + 'RH_null_decoding.txt'
    Model.actflowDecoding(actflow_rh, Model.data_task_rh, rh_filename_null,
                          nbootstraps=nbootstraps, nproc=nproc, null=True, verbose=True)

    print('Run decoding on LH actflow predictions...')
    lh_filename = outdir + 'LH_decoding.txt'
    Model.actflowDecoding(actflow_lh, Model.data_task_lh, lh_filename,
                          nbootstraps=nbootstraps, nproc=nproc, null=False, verbose=True)

    print('Run null (permutation) decoding on LH actflow predictions...')
    lh_filename_null = outdir + 'LH_null_decoding.txt'
    Model.actflowDecoding(actflow_lh, Model.data_task_lh, lh_filename_null,
                          nbootstraps=nbootstraps, nproc=nproc, null=True, verbose=True)

def runRandomizedHiddenLayerModel_v2(projectdir=projectdir,
                                     outdir=projectdir + 'data/results/MAIN/srModelPredictionAccuracies/SRActFlowRandomizedHidden_v2/',
                                     thresh=0,nhidden=10,computeFC=True,ncomponents=500,nproc=10,
                                     scratchfcdir=projectdir + 'data/results/MAIN/fc/LayerToLayerFC_10Hidden_Randomized/',iteration=''):
    """
    For Nat Comms revision: Save out the list of randomized hidden regions for each permutation
    Full SR model with randomized hidden layer
    """
    #### Don't change this parameter -- override
    thresh = 0
    randomize = True
    ####

    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True) # Make sure directory exists
    pathlib.Path(scratchfcdir).mkdir(parents=True, exist_ok=True) # Make sure directory exists

    Model = SRModels.Model(projectdir=projectdir,n_hiddenregions=nhidden,randomize=randomize,scratchfcdir=scratchfcdir)

    #### Only compute FC if it has not been computed before -- computationally intensive process
    if computeFC:
        Model.computeGroupFC(n_components=ncomponents,nproc=nproc)

    #### Load real activations onto model
    Model.loadRealMotorResponseActivations()

    #### Load group FC weights as model weights
    Model.loadModelFC()

    #### Run SRActFlow simulations on entire group
    actflow_rh, actflow_lh = Model.simulateGroupActFlow(thresh=thresh,nproc=nproc)

    #### Evaluate how well actflow performs  relative to actual activations
    nbootstraps = 1
    print('Run decoding on RH actflow predictions...')
    rh_filename = outdir + 'RH_decoding' + str(iteration) + '.txt'
    Model.actflowDecoding(actflow_rh, Model.data_task_rh, rh_filename,
                          nbootstraps=nbootstraps, nproc=nproc, null=False, verbose=True)

    print('Run null (permutation) decoding on RH actflow predictions...')
    rh_filename_null = outdir + 'RH_null_decoding' + str(iteration) + '.txt'
    Model.actflowDecoding(actflow_rh, Model.data_task_rh, rh_filename_null,
                          nbootstraps=nbootstraps, nproc=nproc, null=True, verbose=True)

    print('Run decoding on LH actflow predictions...')
    lh_filename = outdir + 'LH_decoding' + str(iteration) + '.txt'
    Model.actflowDecoding(actflow_lh, Model.data_task_lh, lh_filename,
                          nbootstraps=nbootstraps, nproc=nproc, null=False, verbose=True)

    print('Run null (permutation) decoding on LH actflow predictions...')
    lh_filename_null = outdir + 'LH_null_decoding' + str(iteration) + '.txt'
    Model.actflowDecoding(actflow_lh, Model.data_task_lh, lh_filename_null,
                          nbootstraps=nbootstraps, nproc=nproc, null=True, verbose=True)

    ##### Save out hidden regions used for this permutation
    # Open/create file
    filetxt = open(outdir + 'HiddenRegionList' + str(iteration) + '.txt',"a+b")
    # Write out to file
    parcels = Model.hiddenregions.copy()
    parcels.shape = (1,len(parcels))
    np.savetxt(filetxt,parcels)
    #print(Model.hiddenregions,file=filetxt)
    # Close file
    filetxt.close()

def extractHiddenRSM(projectdir=projectdir,outdir=projectdir+'data/results/MAIN/srModelPredictionAccuracies/hiddenRSMs/',
                   thresh=0,ruletype='12',nhidden=10,computeFC=False,ncomponents=500,nproc=10):
    """
    function that just generates the hidden layer activations with and without a threshold (relu)
    """
    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True) # Make sure directory exists
    filename = outdir + "HiddenActivations"

    Model = SRModels.Model(projectdir=projectdir,ruletype=ruletype,n_hiddenregions=nhidden,randomize=False)

    #### Only compute FC if it has not been computed before -- computationally intensive process
    if computeFC:
        Model.computeGroupFC(n_components=ncomponents,nproc=nproc)

    #### Load group FC weights as model weights
    Model.loadModelFC()

    #### Generate hidden layer activations
    Model.generateHiddenUnitRSMPredictions(thresh=thresh,n_hiddenregions=nhidden,filename=filename,verbose=False)

############
#### run FPN variants (using FPN as task-rule input areas)
def runFullSRModelFPN(projectdir=projectdir,outdir=projectdir+'data/results/MAIN/srModelPredictionAccuracies/SRActFlowFullFPN/',
                      thresh=0,ruletype='fpn',nhidden=10,computeFC=False,ncomponents=500,nproc=10):
    """
    Full SR model 
    """
    #### Don't change this parameter -- override
    thresh = 0
    ####

    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True) # Make sure directory exists

    Model = SRModels.Model(projectdir=projectdir,ruletype=ruletype,n_hiddenregions=nhidden,randomize=False)

    #### Only compute FC if it has not been computed before -- computationally intensive process
    if computeFC:
        Model.computeGroupFC(n_components=ncomponents,nproc=nproc)

    #### Load real activations onto model
    Model.loadRealMotorResponseActivations()

    #### Load group FC weights as model weights
    Model.loadModelFC()

    #### Run SRActFlow simulations on entire group
    actflow_rh, actflow_lh = Model.simulateGroupActFlow(thresh=thresh,nproc=nproc)

    #### Evaluate how well actflow performs  relative to actual activations
    nbootstraps = 1000
    print('Run decoding on RH actflow predictions...')
    rh_filename = outdir + 'RH_decoding.txt'
    Model.actflowDecoding(actflow_rh, Model.data_task_rh, rh_filename,
                          nbootstraps=nbootstraps, nproc=nproc, null=False, verbose=True)

    print('Run null (permutation) decoding on RH actflow predictions...')
    rh_filename_null = outdir + 'RH_null_decoding.txt'
    Model.actflowDecoding(actflow_rh, Model.data_task_rh, rh_filename_null,
                          nbootstraps=nbootstraps, nproc=nproc, null=True, verbose=True)

    print('Run decoding on LH actflow predictions...')
    lh_filename = outdir + 'LH_decoding.txt'
    Model.actflowDecoding(actflow_lh, Model.data_task_lh, lh_filename,
                          nbootstraps=nbootstraps, nproc=nproc, null=False, verbose=True)

    print('Run null (permutation) decoding on LH actflow predictions...')
    lh_filename_null = outdir + 'LH_null_decoding.txt'
    Model.actflowDecoding(actflow_lh, Model.data_task_lh, lh_filename_null,
                          nbootstraps=nbootstraps, nproc=nproc, null=True, verbose=True)

def runFullSRModelNoUnimodal(projectdir=projectdir,outdir=projectdir+'data/results/MAIN/srModelPredictionAccuracies/SRActFlowFullNoUnimodal/',
                      thresh=0,ruletype='nounimodal',nhidden=10,computeFC=False,ncomponents=500,nproc=10):
    """
    Full SR model 
    """
    #### Don't change this parameter -- override
    thresh = 0
    ####

    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True) # Make sure directory exists

    Model = SRModels.Model(projectdir=projectdir,ruletype=ruletype,n_hiddenregions=nhidden,randomize=False)

    #### Only compute FC if it has not been computed before -- computationally intensive process
    if computeFC:
        Model.computeGroupFC(n_components=ncomponents,nproc=nproc)

    #### Load real activations onto model
    Model.loadRealMotorResponseActivations()

    #### Load group FC weights as model weights
    Model.loadModelFC()

    #### Run SRActFlow simulations on entire group
    actflow_rh, actflow_lh = Model.simulateGroupActFlow(thresh=thresh,nproc=nproc)

    #### Evaluate how well actflow performs  relative to actual activations
    nbootstraps = 1000
    print('Run decoding on RH actflow predictions...')
    rh_filename = outdir + 'RH_decoding.txt'
    Model.actflowDecoding(actflow_rh, Model.data_task_rh, rh_filename,
                          nbootstraps=nbootstraps, nproc=nproc, null=False, verbose=True)

    print('Run null (permutation) decoding on RH actflow predictions...')
    rh_filename_null = outdir + 'RH_null_decoding.txt'
    Model.actflowDecoding(actflow_rh, Model.data_task_rh, rh_filename_null,
                          nbootstraps=nbootstraps, nproc=nproc, null=True, verbose=True)

    print('Run decoding on LH actflow predictions...')
    lh_filename = outdir + 'LH_decoding.txt'
    Model.actflowDecoding(actflow_lh, Model.data_task_lh, lh_filename,
                          nbootstraps=nbootstraps, nproc=nproc, null=False, verbose=True)

    print('Run null (permutation) decoding on LH actflow predictions...')
    lh_filename_null = outdir + 'LH_null_decoding.txt'
    Model.actflowDecoding(actflow_lh, Model.data_task_lh, lh_filename_null,
                          nbootstraps=nbootstraps, nproc=nproc, null=True, verbose=True)

### Out of date as of 10/21/21
#def runRandomizedHiddenLayerModel(projectdir=projectdir,outdir=projectdir+'data/results/MAIN/srModelPredictionAccuracies/SRActFlowRandomizedHidden/',
#                   thresh=0,nhidden=10,computeFC=False,ncomponents=500,nproc=10,scratchfcdir=None):
#    """
#    Full SR model 
#    """
#    #### Don't change this parameter -- override
#    thresh = 0
#    randomize = True
#    ####
#
#    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True) # Make sure directory exists
#
#    Model = SRModels.Model(projectdir=projectdir,n_hiddenregions=nhidden,randomize=randomize,scratchfcdir=scratchfcdir)
#
#    #### Only compute FC if it has not been computed before -- computationally intensive process
#    if computeFC:
#        Model.computeGroupFC(n_components=ncomponents,nproc=nproc)
#
#    #### Load real activations onto model
#    Model.loadRealMotorResponseActivations()
#
#    #### Load group FC weights as model weights
#    Model.loadModelFC()
#
#    #### Run SRActFlow simulations on entire group
#    actflow_rh, actflow_lh = Model.simulateGroupActFlow(thresh=thresh,nproc=nproc)
#
#    #### Evaluate how well actflow performs  relative to actual activations
#    nbootstraps = 1
#    print('Run decoding on RH actflow predictions...')
#    rh_filename = outdir + 'RH_decoding.txt'
#    Model.actflowDecoding(actflow_rh, Model.data_task_rh, rh_filename,
#                          nbootstraps=nbootstraps, nproc=nproc, null=False, verbose=True)
#
#    print('Run null (permutation) decoding on RH actflow predictions...')
#    rh_filename_null = outdir + 'RH_null_decoding.txt'
#    Model.actflowDecoding(actflow_rh, Model.data_task_rh, rh_filename_null,
#                          nbootstraps=nbootstraps, nproc=nproc, null=True, verbose=True)
#
#    print('Run decoding on LH actflow predictions...')
#    lh_filename = outdir + 'LH_decoding.txt'
#    Model.actflowDecoding(actflow_lh, Model.data_task_lh, lh_filename,
#                          nbootstraps=nbootstraps, nproc=nproc, null=False, verbose=True)
#
#    print('Run null (permutation) decoding on LH actflow predictions...')
#    lh_filename_null = outdir + 'LH_null_decoding.txt'
#    Model.actflowDecoding(actflow_lh, Model.data_task_lh, lh_filename_null,
#                          nbootstraps=nbootstraps, nproc=nproc, null=True, verbose=True)


if __name__ == '__main__':
    args = parser.parse_args()
    run(args)
