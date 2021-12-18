import numpy as np
np.set_printoptions(suppress=True)
#import matplotlib.pyplot as plt
#import seaborn as sns
#import scipy.stats as stats
import os
os.sys.path.append('model/')
import model.model_nodynamics as model
import model.task as task
import time
import model.analysis as analysis
import argparse
#import torch
#from torch.autograd import Variable
#import torch.nn.functional as F

### Sample command
# run.runModel(num_hidden=1280,training=True,save_csv=True,niterations=1) 
## For random initialization (null):
# run.runModel(num_hidden=1280,training=False,save_csv=True,niterations=1000) # randomly initlize 1000 times and compute RSM of hidden layer 


projectdir = '/home/ti61/f_mc1689_1/SRActFlow/'
projectdir = '../../'

parser = argparse.ArgumentParser('./main.py', description='Run RSM analysis for each parcel using vertex-wise activations')
parser.add_argument('--num_hidden', type=int, default=1280, help='# hidden units (default: 1280)')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate (default: 0.0001)')
parser.add_argument('--thresh', type=float, default=0.9, help='threshold for classifying output units (default: 0.9)')
parser.add_argument('--create_new_batches', action='store_true', help="Create new training batches. Mostly not necessary, since already exist in data directory")
parser.add_argument('--save_csv', action='store_true', help="Save out the RSM?")
parser.add_argument('--save_hiddenrsm_pdf', action='store_true', help="Save out a pdf of the RSM?")
parser.add_argument('--shuffledFC', action='store_true', help="shuffle fc after training (null model)")
parser.add_argument('--shuffledLabels', action='store_true', help="shuffle training labels?")
parser.add_argument('--training', action='store_true', help="train model (if not, no training -> null model")
parser.add_argument('--trackloss', action='store_true', help="track loss during training")
parser.add_argument('--niterations', type=int, default=1, help='# iterations (useful for null models) (default: 1)')
parser.add_argument('--batchstop', type=int, default=0, help='# batches to stop (early stopping) (default: 0)')

#def runModel(num_hidden=1280,learning_rate=0.0001,thresh=0.9,create_new_batches=False,save_csv=False,save_hiddenrsm_pdf=False,shuffledFC=False,shuffledLabels=False,training=True,niterations=1):
def run(args):
    """
    num_hidden - # of hidden units
    learning_rate - learning rate 
    thresh - threshold for classifying output units
    create_new_batches - Create new training batches. Most likely not necessary; training batches already exist in data directory
    save_csv - Save out the RSM?
    save_hiddenrsm_pdf - save out a PDF of the RSM?
    """
    args 
    num_hidden = args.num_hidden
    learning_rate = args.learning_rate
    thresh = args.thresh
    create_new_batches = args.create_new_batches
    save_csv = args.save_csv
    save_hiddenrsm_pdf = args.save_hiddenrsm_pdf
    shuffledFC = args.shuffledFC
    shuffledLabels = args.shuffledLabels
    training = args.training
    niterations = args.niterations
    trackloss = args.trackloss
    batchstop = args.batchstop
    if batchstop == 0: batchstop = None

    if create_new_batches: 
        TrialInfo = model.TrialBatches(NUM_BATCHES=30000,
                                       NUM_TASKS_IN_TRAINSET=64,
                                       NUM_TASK_TYPES_PER_BATCH=64,
                                       NUM_TESTING_TRIAlS_PER_TASK=100,
                                       NUM_TRAINING_TRIAlS_PER_TASK=10,
                                       NUM_INPUT_ELEMENTS=28,
                                       NUM_OUTPUT_ELEMENTS=4,
                                       filename=projectdir + 'data/results/MODEL/TrialBatches_Default_NoDynamics_new')
        TrialInfo.createAllBatches(nproc=30)

    #### Load training batches
    print('Loading training batches')
    input_batches, output_batches = model.load_training_batches(cuda=False,filename=projectdir + 'data/results/MODEL/TrialBatches_Default_NoDynamics')

    rsms = []
    for i in range(niterations):
        Network = model.ANN(num_rule_inputs=12,
                             num_sensory_inputs=16,
                             num_hidden=num_hidden,
                             num_motor_decision_outputs=4,
                             learning_rate=learning_rate,
                             thresh=thresh)
        # Network.cuda = True
        Network = Network.cpu()

        if shuffledLabels:
            ntrials_per_epoch = output_batches.shape[1]
            shuffle_ind = np.arange(ntrials_per_epoch)
            np.random.shuffle(shuffle_ind)
            output_batches = output_batches[:,shuffle_ind,:]

        if training:
            #### Train model
            print('Training model')
            timestart = time.time()
            if trackloss:
                train_loss = model.batch_training(Network, input_batches,output_batches,track_loss=trackloss,batchstop=batchstop,cuda=False)  
            else:
                model.batch_training(Network, input_batches,output_batches,track_loss=trackloss,batchstop=batchstop,cuda=False)  
            timeend = time.time()
            print('Time elapsed using CPU:', timeend-timestart)

        #### Save out hidden layer RSM
        if shuffledFC:
            hidden, rsm = analysis.rsa_shuffleconn(Network,show=save_hiddenrsm_pdf,savepdf=save_hiddenrsm_pdf,nshuffle=10000)
        else:
            hidden, rsm = analysis.rsa(Network,show=save_hiddenrsm_pdf,savepdf=save_hiddenrsm_pdf)
        # hidden = hidden.detach().numpy()
        # input_matrix = input_matrix.detach().numpy()

        rsms.append(rsm)
    
    rsm = np.mean(np.asarray(rsms),axis=0)

    # Save out RSM 
    if save_csv:
        shuffle = '_ShuffledConn' if shuffledFC else ''
        shuffle = shuffle + '_ShuffledLabels' if shuffledLabels else shuffle + ''
        randominit = '_randomInit' if not training else ''
        outputfilename = 'ANN' + str(num_hidden) + '_HiddenLayerRSM_NoDynamics' + shuffle + randominit
        np.savetxt(outputfilename + '.csv',rsm)
        if trackloss: np.savetxt(outputfilename + '_trainingloss.csv',np.asarray(train_loss))


if __name__ == '__main__':
    args = parser.parse_args()
    run(args)
