import numpy as np
import multiprocessing as mp
import scipy.stats as stats
import os
os.environ['OMP_NUM_THREADS'] = str(1)
import h5py
import tools_group_rsa as tools_group
import nibabel as nib
import EmpiricalSRActFlow_ANN_RSA as esr
import time
import tools_group_rsa as tools_group
tools_group = reload(tools_group)
esr = reload(esr)

# Excluding 084
subjNums = ['013','014','016','017','018','021','023','024','026','027','028','030','031','032','033',
'034','035','037','038','039','040','041','042','043','045','046','047','048','049','050',
'053','055','056','057','058','062','063','066','067','068','069','070','072','074','075',
'076','077','081','085','086','087','088','090','092','093','094','095','097','098','099',
'101','102','103','104','105','106','108','109','110','111','112','114','115','117','119',
'120','121','122','123','124','125','126','127','128','129','130','131','132','134','135',
'136','137','138','139','140','141']



def main(nproc=6,outdir='/projects3/SRActFlow/data/results/GroupfMRI/GroupfMRI10c/'):
    timestart = time.time()

    ###################################
    runSRActFlow(nproc=6,outdir=outdir)
    ###################################

    timeend = time.time()
    print('Total time elapsed: ' + str(timeend-timestart))

def runSRActFlow(nproc=6,outdir='/projects3/SRActFlow/data/results/GroupfMRI/GroupfMRI10c/'):
    print('Running SRActFlow, excluding task context')
    print 'Load FC'
    global fc_input2hidden
    global fc_logic2hidden
    global fc_sensory2hidden
    global fc_motor2hidden
    global fc_hidden2motorresp
    inputtypes = ['color','ori','pitch','constant']
    inputkeys = ['RED','VERTICAL','HIGH','CONSTANT']
    fc_input2hidden = {}
    i = 0
    for inputtype in inputtypes:
        fc_input2hidden[inputkeys[i]] = np.real(tools_group.loadGroupActFlowFC(inputtype))
        i += 1

    # Load rules to hidden FC mappings
    fc_logic2hidden = np.real(tools_group.loadGroupActFlowFC('Logic'))
    fc_sensory2hidden = np.real(tools_group.loadGroupActFlowFC('Sensory'))
    fc_motor2hidden = np.real(tools_group.loadGroupActFlowFC('Motor'))

    # Load hidden to motor resp mappings
    fc_hidden2motorresp = np.real(tools_group.loadGroupActFlowFC('hidden2out'))

    #threshold FC
    # for key in fc_input2hidden: 
    #     fc_input2hidden[key] = np.multiply(fc_input2hidden[key]>0,fc_input2hidden[key])

    # threshold FC
    fc_hidden2motorresp = np.multiply(fc_hidden2motorresp>0,fc_hidden2motorresp)

    # multiprocessing
    timestart = time.time()
    pool = mp.Pool(processes=nproc)
    results = pool.map_async(subjSRActFlow,subjNums).get()
    pool.close()
    pool.join()
    timeend = time.time()
    print "time elapsed no context:", timeend-timestart

    actflow_predictions = np.asarray(results)

    # Save output
    h5f = h5py.File(outdir + 'GroupfMRI10c.h5','a')
    try:
        h5f.create_dataset('actflow_predictions',data=actflow_predictions)
    except:
        del h5f['actflow_predictions']
        h5f.create_dataset('actflow_predictions',data=actflow_predictions)
    h5f.close()

def subjSRActFlow(subj):
    """
    Local function for subj-level SRActFlow
    """
    print 'Subject', subj
    obj = esr.EmpiricalActFlow(subj)
    # Input
    obj.fc_input2hidden = fc_input2hidden
    # Rules
    obj.fc_logic2hidden = fc_logic2hidden
    obj.fc_sensory2hidden = fc_sensory2hidden
    obj.fc_motor2hidden = fc_motor2hidden
    # hidden 2 motor
    obj.fc_hidden2motorresp = fc_hidden2motorresp


    actflow = obj.generateActFlowPredictions(verbose=False, n_inputs=1)
    del obj
    return actflow
