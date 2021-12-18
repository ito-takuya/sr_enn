import numpy as np
import multiprocessing as mp
import scipy.stats as stats
import os
os.environ['OMP_NUM_THREADS'] = str(1)
import statsmodels.sandbox.stats.multicomp as mc
import h5py
import nibabel as nib
from importlib import reload
import tools
import argparse


# Excluding 084
subjNums = ['013','014','016','017','018','021','023','024','026','027','028','030','031','032','033',
            '034','035','037','038','039','040','041','042','043','045','046','047','048','049','050',
            '053','055','056','057','058','062','063','066','067','068','069','070','072','074','075',
            '076','077','081','085','086','087','088','090','092','093','094','095','097','098','099',
            '101','102','103','104','105','106','108','109','110','111','112','114','115','117','119',
            '120','121','122','123','124','125','126','127','128','129','130','131','132','134','135',
            '136','137','138','139','140','141']



projectdir = '/home/ti61/f_mc1689_1/SRActFlow/'

# Using final partition
networkdef = np.loadtxt(projectdir + 'data/network_partition.txt')
networkorder = np.asarray(sorted(range(len(networkdef)), key=lambda k: networkdef[k]))
networkorder.shape = (len(networkorder),1)
# network mappings for final partition set
networkmappings = {'fpn':7, 'vis1':1, 'vis2':2, 'smn':3, 'aud':8, 'lan':6, 'dan':5, 'con':4, 'dmn':9, 
                   'pmulti':10, 'none1':11, 'none2':12}
networks = networkmappings.keys()

xticks = {}
reorderednetworkaffil = networkdef[networkorder]
for net in networks:
    netNum = networkmappings[net]
    netind = np.where(reorderednetworkaffil==netNum)[0]
    tick = np.max(netind)
    xticks[tick] = net

## General parameters/variables
nParcels = 360
nSubjs = len(subjNums)

glasserfile2 = projectdir + 'data/Q1-Q6_RelatedParcellation210.LR.CorticalAreas_dil_Colors.32k_fs_RL.dlabel.nii'
glasser2 = nib.load(glasserfile2).get_data()
glasser2 = np.squeeze(glasser2)

sortednets = np.sort(list(xticks.keys()))
orderednetworks = []
for net in sortednets: orderednetworks.append(xticks[net])
    
networkpalette = ['royalblue','slateblue','paleturquoise','darkorchid','limegreen',
                  'lightseagreen','yellow','orchid','r','peru','orange','olivedrab']
networkpalette = np.asarray(networkpalette)

OrderedNetworks = ['VIS1','VIS2','SMN','CON','DAN','LAN','FPN','AUD','DMN','PMM','VMM','ORA']


parser = argparse.ArgumentParser('./main.py', description='Run decoding analyses to identify sets of regions involved for different task components')
parser.add_argument('--outfilename', type=str, default="", help='Prefix output filenames (Default: analysis1')
parser.add_argument('--decoder', type=str, default="similarity", help='decoding approach [similarity, logistic, svm')
parser.add_argument('--ID', type=str, default="rule", help='condition/information to decode [rules,colorStim,oriStim,constantStim,pitchStim]')
parser.add_argument('--nproc', type=int, default=20, help='number of processes to run in parallel (default: 20)')
parser.add_argument('--motor_mapping', action='store_true', help="Include motor output activations")

def run(args):
    args
    outfilename = args.outfilename
    decoder = args.decoder
    ID = args.ID
    nproc = args.nproc
    outdir = '/home/ti61/f_mc1689_1/SRActFlow/data/results/MAIN/LayerID_Revision/'
    outfilename = outdir + ID + '_' + decoder
    

    

    if ID=='rules':
        nStims = 12
        data_task = np.zeros((len(glasser2),nStims,len(subjNums)))
        rules = ['Logic','Sensory','Motor']
        rois = np.arange(nParcels)

        scount = 0
        for subj in subjNums:
            rulecount = 0
            for rule in rules:
                data_task[:,rulecount:(rulecount+4),scount] = tools.loadInputActivity(subj,rule)
                rulecount += 4
            scount += 1

    if ID in ['colorStim', 'oriStim', 'constantStim', 'pitchStim']:
        nStims = 4
        data_task = np.zeros((len(glasser2),nStims,len(subjNums)))

        if ID in ['colorStim','oriStim']:
            rois = np.where((networkdef==networkmappings['vis1']) | (networkdef==networkmappings['vis2']))[0] 
        elif ID in ['constantStim','pitchStim']:
            rois = np.where(networkdef==networkmappings['aud'])[0]

        scount = 0
        for subj in subjNums:
            data_task[:,:,scount] = tools.loadInputActivity(subj,ID)
            scount += 1

    distances_baseline_allrules, rmatch, rmismatch, confusion_mats = tools.conditionDecodings(data_task, rois, 
                                                                                              motorOutput=False, ncvs=1, effects=True, 
                                                                                              confusion=True, decoder=decoder, nproc=nproc)


    statistics_allrules = np.zeros((len(rois),3)) # acc, q, acc_thresh
    for roicount in range(len(rois)):
        ntrials = distances_baseline_allrules.shape[1]
        p = stats.binom_test(np.mean(distances_baseline_allrules[roicount,:])*ntrials,n=ntrials,p=1/float(nStims))
        if np.mean(distances_baseline_allrules[roicount,:])>1/float(nStims):
            p = p/2.0
        else:
            p = 1.0-p/2.0
            

        statistics_allrules[roicount,0] = np.mean(distances_baseline_allrules[roicount,:])
        statistics_allrules[roicount,1] = p

    h0, qs = mc.fdrcorrection0(statistics_allrules[:,1])
    for roicount in range(len(rois)):
        statistics_allrules[roicount,1] = qs[roicount]
        statistics_allrules[roicount,2] = h0[roicount]*statistics_allrules[roicount,0]
        
    # Count number of significant ROIs for LH decoding
    sig_ind = np.where(statistics_allrules[:,1]<0.05)[0]
    print('Number of ROIs significant for all 12 rules:', sig_ind.shape[0])
    print('Accuracies:', statistics_allrules[sig_ind,0])

    #### Map back to surface
    # Put all data into a single matrix (since we only run a single classification)
    inputStim = np.zeros((glasser2.shape[0],3))

    roicount = 0
    for roi in rois:
        vertex_ind = np.where(glasser2==roi+1)[0]
        inputStim[vertex_ind,0] = statistics_allrules[roicount,0]
        inputStim[vertex_ind,1] = statistics_allrules[roicount,1]
        inputStim[vertex_ind,2] = statistics_allrules[roicount,2]

        roicount += 1

    np.savetxt(outfilename + '.csv', np.where(statistics_allrules[:,1]<0.05)[0], delimiter=',')
        
    #### 
    # Write file to csv and run wb_command
    np.savetxt(outfilename + '.csv', inputStim,fmt='%s')
    wb_command = 'wb_command -cifti-convert -from-text ' + outfilename + '.csv ' + glasserfile2 + ' ' + outfilename + '.dscalar.nii' + ' -reset-scalars'
    os.system(wb_command)

if __name__ == '__main__':
    args = parser.parse_args()
    run(args)
