# Taku Ito
# 05/10/2019
# RNN model training with no trial dynamics
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
np.set_printoptions(suppress=True)
import time


basedir = '/home/ti61/f_mc1689_1/SRActFlow/'

class RNN(torch.nn.Module):
    """
    Neural network object
    """

    def __init__(self,
                 num_rule_inputs=12,
                 num_sensory_inputs=16,
                 hidden_width=128,
                 n_hidden_layers=1,
                 num_motor_decision_outputs=4,
                 learning_rate=0.0001,
                 thresh=0.8,
                 cuda=False):

        # Define general parameters
        self.num_rule_inputs = num_rule_inputs
        self.num_sensory_inputs =  num_sensory_inputs
        self.hidden_width = hidden_width
        self.n_hidden_layers = n_hidden_layers
        self.num_motor_decision_outputs = num_motor_decision_outputs
        self.cuda = cuda

        # Define entwork architectural parameters
        super(RNN,self).__init__()

        self.w_in = torch.nn.Linear(num_sensory_inputs+num_rule_inputs,hidden_width)
        self.w_rec = torch.nn.Linear(hidden_width,hidden_width)
        self.w_out = torch.nn.Linear(hidden_width,num_motor_decision_outputs)
        self.sigmoid = torch.nn.Sigmoid()
        self.func = torch.nn.ReLU()

        # Initialize tensors (don't need to initialize -- intialized from uniform dist)
        #torch.nn.init.normal_(self.w_in, mean=0.0, std=1.0)
        #torch.nn.init.normal_(self.w_rec, mean=0.0, std=1.0)
        #torch.nn.init.normal_(self.w_out, mean=0.0, std=1.0)

        # Define loss function
        self.lossfunc = torch.nn.MSELoss(reduce=False)

        # Decision threshhold for behavior
        self.thresh = thresh

        # Construct optimizer
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        ##optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
    
    def initHidden(self):
        return torch.randn(1, self.hidden_width)

    def forward(self,inputs,noise=False):
        """
        Run a forward pass of a trial by input_elements matrix
        """
        if self.n_hidden_layers<1: 
            raise Exception("ERROR -- no hidden layers")

        hidden_layers = np.zeros((inputs.shape[0],self.hidden_width,self.n_hidden_layers))
        for layer in range(self.n_hidden_layers):
            if layer==0:
                # Map inputs into RNN space
                hidden = self.w_in(inputs)
                hidden = self.func(hidden)
            else:
                # Run RNN
                hidden = self.w_rec(hidden)
                hidden = self.func(hidden)

            # Define rnn private noise/spont_act
            if noise:
                spont_act = torch.randn(hidden.shape, dtype=torch.float)/np.sqrt(self.hidden_width)
                # Add private noise to each unit, and add to the input
                hidden = hidden + spont_act

            hidden_layers[:,:,layer] = hidden.detach().numpy()

        # Compute outputs
        h2o = self.w_out(hidden) # Generate linear outupts
        outputs = self.sigmoid(h2o) # Pass through nonlinearity

        return outputs, hidden_layers


def train(network, inputs, targets):
    """Train network"""
    network.train()
    network.zero_grad()
    network.optimizer.zero_grad()

    outputs, hidden = network.forward(inputs,noise=False)

    # Calculate loss
    loss = network.lossfunc(outputs,targets)
    loss = torch.mean(loss)
    
    # Backprop and update weights
    loss.backward()
    network.optimizer.step() # Update parameters using optimizer

    return outputs, targets, loss

def batch_training(network,train_inputs,train_outputs,
                     cuda=False):

    accuracy_per_batch = []

    batch_ordering = np.arange(train_inputs.shape[0])

    np.random.shuffle(batch_ordering)

    timestart = time.time()
    for batch in np.arange(train_inputs.shape[0]):
        batch_id = batch_ordering[batch]

        if cuda:
            train_input_batch = train_inputs[batch_id,:,:].cuda()
            train_output_batch = train_outputs[batch_id,:,:].cuda()
        else:
            train_input_batch = train_inputs[batch_id,:,:]
            train_output_batch = train_outputs[batch_id,:,:]

        outputs, targets, loss = train(network,
                                       train_inputs[batch_id,:,:],
                                       train_outputs[batch_id,:,:])

        if batch % 5000 == 0:
            targets = targets.cpu()
            outputs = outputs.cpu()
    
            acc = [] # accuracy array
            for mb in range(targets.shape[0]):
                for out in range(targets.shape[1]):
                    if targets[mb,out] == 0: continue
                    response = outputs[mb,out] # Identify response time points
                    thresh = network.thresh # decision thresh
                    target_resp = torch.ByteTensor([out]) # The correct target response
                    max_resp = outputs[mb,:].argmax().byte()
                    if max_resp==target_resp and response>thresh: # Make sure network response is correct respnose, and that it exceeds some threshold
                        acc.append(1.0)
                    else:
                        acc.append(0)

            timeend = time.time()
            print('Iteration:', batch)
            print('\tloss:', loss.item())
            print('Time elapsed...', timeend-timestart)
            timestart = timeend
            print('\tAccuracy: ', str(round(np.mean(acc)*100.0,4)),'%')
        
            nbatches_break = 1000
            if batch>nbatches_break:
                if np.sum(np.asarray(accuracy_per_batch[-nbatches_break:])>99.0)==nbatches_break:
                    print('Last', nbatches_break, 'batches had above 99.5% accuracy... stopping training')
                    break

        accuracy_per_batch.append(np.mean(acc)*100.0)

def eval(network,test_inputs,targets,cuda=False):
    network.eval()
    network.zero_grad()
    network.optimizer.zero_grad()

    outputs, hidden = network.forward(test_inputs,noise=True)

    # Calculate loss
    loss = network.lossfunc(outputs,targets)
    loss = torch.mean(loss)


    acc = [] # accuracy array
    for mb in range(targets.shape[0]):
        for out in range(targets.shape[1]):
            if targets[mb,out] == 0: continue
            response = outputs[mb,out] # Identify response time points
            thresh = network.thresh # decision thresh
            target_resp = torch.ByteTensor([out]) # The correct target response
            max_resp = outputs[mb,:].argmax().byte()
            if max_resp==target_resp and response>thresh: # Make sure network response is correct respnose, and that it exceeds some threshold
                acc.append(1.0)
            else:
                acc.append(0)

    print('\tloss:', loss.item())
    print('\tAccuracy: ',str(round(np.mean(acc)*100.0,4)),'%')
    return outputs, hidden

def load_training_batches(cuda=False,filename='/home/ti61/f_mc1689_1/SRActFlow/data/results/MODEL/TrialBatches_Default_NoDynamics'):
    TrialObj = TrialBatches(filename=filename)
    inputs, outputs = TrialObj.loadTrainingBatches()

    inputs = inputs.float()
    outputs = outputs.float()
    if cuda==True:
        inputs = inputs.cuda()
        outputs = outputs.cuda()

    return inputs, outputs

def load_testset(cuda=False,filename='/home/ti61/f_mc1689_1/SRActFlow/data/results/MODEL/TrialBatches_Default_NoDynamics'):
    TrialObj = TrialBatches(filename=filename)
    test_inputs, test_outputs = TrialObj.loadTestset()

    test_inputs = test_inputs.float()
    test_outputs = test_outputs.float()
    if cuda==True:
        test_inputs = test_inputs.cuda()
        test_outputs = test_outputs.cuda()
    return test_inputs, test_outputs

class TrialBatches(object):
    """
    Batch trials
    """
    def __init__(self,
                 NUM_BATCHES=100000,
                 NUM_TASKS_IN_TRAINSET=63,
                 NUM_TASK_TYPES_PER_BATCH=63,
                 NUM_TESTING_TRIAlS_PER_TASK=100,
                 NUM_TRAINING_TRIAlS_PER_TASK=1,
                 NUM_INPUT_ELEMENTS=28,
                 NUM_OUTPUT_ELEMENTS=4,
                 filename='/home/ti61/f_mc1689_1/SRActFlow/data/results/MODEL/TrialBatches_Default_NoDynamics'):


        self.NUM_BATCHES = NUM_BATCHES
        self.NUM_TASKS_IN_TRAINSET = NUM_TASKS_IN_TRAINSET
        self.NUM_TASK_TYPES_PER_BATCH = NUM_TASK_TYPES_PER_BATCH
        self.NUM_OUTPUT_ELEMENTS = NUM_OUTPUT_ELEMENTS
        self.NUM_TESTING_TRIAlS_PER_TASK = NUM_TESTING_TRIAlS_PER_TASK
        self.NUM_TRAINING_TRIAlS_PER_TASK = NUM_TRAINING_TRIAlS_PER_TASK
        self.NUM_INPUT_ELEMENTS = NUM_INPUT_ELEMENTS
        self.splitTrainTestTaskSets(n_trainset=NUM_TASKS_IN_TRAINSET)
        self.filename = filename

    def createAllBatches(self,nproc=10):
        # Initialize empty tensor for batches
        num_trials = self.NUM_TASK_TYPES_PER_BATCH 
        
        batch_inputtensor = np.zeros((self.NUM_INPUT_ELEMENTS, len(self.trainRuleSet)*self.NUM_TRAINING_TRIAlS_PER_TASK, self.NUM_BATCHES))
        batch_outputtensor = np.zeros((self.NUM_OUTPUT_ELEMENTS, len(self.trainRuleSet)*self.NUM_TRAINING_TRIAlS_PER_TASK, self.NUM_BATCHES))

        inputs = []
        for batch in range(self.NUM_BATCHES):
            shuffle = True
            seed = np.random.randint(1000000)
            inputs.append((self.trainRuleSet,self.NUM_TRAINING_TRIAlS_PER_TASK,shuffle,batch,seed))

        pool = mp.Pool(processes=nproc)
        results = pool.starmap_async(create_trial_batches,inputs).get()
        pool.close()
        pool.join()

        batch = 0
        for result in results:
            batch_inputtensor[:,:,batch] = result[0]
            batch_outputtensor[:,:,batch] = result[1]
            batch += 1

        # Construct test set
        #nTasks = 64 - self.NUM_TASKS_IN_TRAINSET
        test_inputs, test_targets = create_trial_batches(self.testRuleSet,self.NUM_TESTING_TRIAlS_PER_TASK,shuffle,1,np.random.randint(10000000))

        h5f = h5py.File(self.filename + '.h5','a')
        try:
            h5f.create_dataset('training/inputs',data=batch_inputtensor)
            h5f.create_dataset('training/outputs',data=batch_outputtensor)
            h5f.create_dataset('test/inputs',data=test_inputs)
            h5f.create_dataset('test/outputs',data=test_targets)
        except:
            del h5f['training/inputs'], h5f['training/outputs'], h5f['test/inputs'], h5f['test/outputs']
            h5f.create_dataset('training/inputs',data=batch_inputtensor)
            h5f.create_dataset('training/outputs',data=batch_outputtensor)
            h5f.create_dataset('test/inputs',data=test_inputs)
            h5f.create_dataset('test/outputs',data=test_targets)
        h5f.close()

    def loadTrainingBatches(self):
        h5f = h5py.File(self.filename + '.h5','r')
        inputs = h5f['training/inputs'][:].copy()
        outputs = h5f['training/outputs'][:].copy()
        h5f.close()

        # Input dimensions: input features, nMiniblocks, nBatches
        inputs = np.transpose(inputs, (2, 1, 0)) # convert to: nBatches, nMiniblocks, input dimensions
        outputs = np.transpose(outputs, (2, 1, 0)) # convert to: nBatches, nMiniblocks, input dimensions

        inputs = torch.from_numpy(inputs)
        outputs = torch.from_numpy(outputs)
        return inputs, outputs
    
    def loadTestset(self):
        h5f = h5py.File(self.filename + '.h5','r')
        inputs = h5f['test/inputs'][:].copy()
        outputs = h5f['test/outputs'][:].copy()
        h5f.close()

        # Input dimensions: input features, nMiniblocks
        inputs = np.transpose(inputs, (1, 0)) # convert to: nMiniblocks, input dimensions
        outputs = np.transpose(outputs, (1, 0)) # convert to:  nMiniblocks, input dimensions

        inputs = torch.from_numpy(inputs)
        outputs = torch.from_numpy(outputs)
        return inputs, outputs

    def splitTrainTestTaskSets(self,n_trainset=32):
        self.n_trainset = n_trainset
        self.n_testset = 64 - n_trainset

        taskRuleSet = task.createRulePermutations()
        trainRuleSet, testRuleSet = task.createTrainTestTaskRules(taskRuleSet,nTrainSet=n_trainset, nTestSet=self.n_testset)

        self.taskRuleSet = taskRuleSet
        self.trainRuleSet = trainRuleSet
        self.testRuleSet = testRuleSet

def create_trial_batches(taskRuleSet,ntrials_per_task,shuffle,batchNum,seed):
    """
    Randomly generates a set of stimuli (nStimuli) for each task rule
    Will end up with 64 (task rules) * nStimuli total number of input stimuli
    
    If shuffle keyword is True, will randomly shuffle the training set
    Otherwise will start with taskrule1 (nStimuli), taskrule2 (nStimuli), etc.
    """
    # instantiate random seed (since this is function called in parallel)
    np.random.seed(seed)

    if batchNum%100==0:
        print('Running batch', batchNum)
    
    stimuliSet = task.createSensoryInputs()

    # Create 1d array to randomly sample indices from
    stimIndices = np.arange(len(stimuliSet))
    taskIndices = np.arange(len(taskRuleSet))

    shuffle=True
    

    #randomTaskIndices = np.random.choice(taskIndices,len(taskIndices),replace=False)
    #randomTaskIndices = np.random.choice(taskIndices,nTasks,replace=False)
    #taskRuleSet2 = taskRuleSet.iloc[randomTaskIndices].copy(deep=True)
    #taskRuleSet = taskRuleSet.reset_index(drop=True)
    taskRuleSet = taskRuleSet.reset_index(drop=False)
    #taskRuleSet = taskRuleSet2.copy(deep=True)

    ntrials_total = ntrials_per_task * len(taskRuleSet)
    ####
    # Construct trial dynamics
    rule_ind = np.arange(12) # rules are the first 12 indices of input vector
    stim_ind = np.arange(12,28) # stimuli are the last 16 indices of input vector
    input_size = len(rule_ind) + len(stim_ind)
    input_matrix = np.zeros((input_size,ntrials_total))
    output_matrix = np.zeros((4,ntrials_total))
    trialcount = 0
    for tasknum in range(len(taskRuleSet)):
        

        for i in range(ntrials_per_task):
            rand_stim_ind = np.random.choice(stimIndices,1,replace=False)
            stimuliSet2 = stimuliSet.iloc[rand_stim_ind].copy(deep=True)
            stimuliSet2 = stimuliSet2.reset_index(drop=True)
        
            ## Create trial array
            # Find input code for this task set
            input_matrix[rule_ind,trialcount] = taskRuleSet.Code[tasknum] 
            # Solve task to get the output code
            tmpresp, out_code = task.solveInputs(taskRuleSet.iloc[tasknum], stimuliSet2.iloc[0])

            input_matrix[stim_ind,trialcount] = stimuliSet2.Code[0]
            output_matrix[:,trialcount] = out_code

            trialcount += 1
            
    if shuffle:
        ind = np.arange(input_matrix.shape[1],dtype=int)
        np.random.shuffle(ind)
        input_matrix = input_matrix[:,ind]
        output_matrix = output_matrix[:,ind]
        
    return input_matrix, output_matrix 

