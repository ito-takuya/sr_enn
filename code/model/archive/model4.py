# Taku Ito
# 03/28/2019
# RNN model training with trial dynamics
import pandas as pd
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import task
import multiprocessing as mp
import h5py
task = reload(task)
np.set_printoptions(suppress=True)
import time




class RNN(torch.nn.RNN):
    """
    Neural network object
    """
    def __init__(self,
                 num_rule_inputs=12,
                 num_sensory_inputs=16,
                 num_hidden=128,
                 num_motor_decision_outputs=4,
                 learning_rate=0.0001,
                 thresh=0.8,
                 cuda=False):

        # Define general parameters
        self.num_rule_inputs = num_rule_inputs
        self.num_sensory_inputs =  num_sensory_inputs
        self.num_hidden = num_hidden
        self.num_motor_decision_outputs = num_motor_decision_outputs
        self.cuda = cuda

        # Define entwork architectural parameters
        super(RNN,self).__init__(input_size=self.num_hidden,
                                 hidden_size=self.num_hidden,
                                 num_layers=1,
                                 nonlinearity='relu',
                                 bias=True,
                                 batch_first=False)

        self.w_in = torch.nn.Linear(num_sensory_inputs+num_rule_inputs,num_hidden)
        self.w_out = torch.nn.Linear(num_hidden,num_motor_decision_outputs)
        self.sigmoid = torch.nn.Sigmoid()

        # Define loss function
        self.lossfunc = torch.nn.MSELoss(reduce=False)
        self.m_ij = 3 # Multiplier to weight response periods

        # Decision threshhold for behavior
        self.thresh = thresh

        # Construct optimizer
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        ##optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)


def train(network, inputs, targets):
    """Train network"""
    network.train()
    network.zero_grad()
    network.optimizer.zero_grad()

    # Map inputs into RNN space
    rnn_input = network.w_in(inputs)
    # Define rnn private noise/spont_act
    spont_act = torch.randn(rnn_input.shape, dtype=torch.float)/np.sqrt(network.num_hidden)
    # Add private noise to each unit, and add to the input
    rnn_input = rnn_input + spont_act

    # Run RNN
    outputs, hidden = network(rnn_input)
    
    # Compute outputs
    h2o = network.w_out(outputs) # Generate linear outupts
    outputs = network.sigmoid(h2o) # Pass through nonlinearity

    # Calculate loss
    M = targets*network.m_ij + 1 # Weighted mean matrix
    loss = network.lossfunc(outputs,targets)
    loss = loss * M
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
            train_input_batch = train_inputs[batch_id,:,:,:].cuda()
            train_output_batch = train_outputs[batch_id,:,:,:].cuda()
        else:
            train_input_batch = train_inputs[batch_id,:,:,:]
            train_output_batch = train_outputs[batch_id,:,:,:]

        outputs, targets, loss = train(network,
                                       train_inputs[batch_id,:,:,:],
                                       train_outputs[batch_id,:,:,:])

        if batch % 1000 == 0:
            targets = targets.cpu()
            outputs = outputs.cpu()
    
            acc = [] # accuracy array
            for mb in range(targets.shape[1]):
                for out in range(targets.shape[2]):
                    target_ind = targets[:,mb,out] == 1 
                    if torch.sum(target_ind)==0: continue # If no responses for this finger continue
                    target_ind = target_ind.byte() # identify response time points
                    actual_response = outputs[:,mb,out] # Identify response time points
                    response_periods = torch.nonzero(target_ind)
                    thresh = network.thresh # decision thresh
                    i = 0
                    while i < len(response_periods): 
                        tp = response_periods[i]
                        target_resp = torch.ByteTensor([out]) # The correct target response
                        response = torch.mean(actual_response[tp:tp+2])
                        max_resp = torch.mean(outputs[tp:tp+2,mb,:],dim=0).argmax().byte()
                        if max_resp==target_resp and response>thresh: # Make sure network response is correct respnose, and that it exceeds some threshold
                            acc.append(1.0)
                        else:
                            acc.append(0)
                        i += 2 # Each response period is 2 time points long, so go to next response period

            timeend = time.time()
            print 'Iteration:', batch
            print '\tloss:', loss.item()
            print 'Time elapsed...', timeend-timestart
            timestart = timeend
            print '\tAccuracy: ' + str(round(np.mean(acc)*100.0,4)) +'%'
        
            nbatches_break = 1000
            if batch>nbatches_break:
                if np.sum(np.asarray(accuracy_per_batch[-nbatches_break:])>99.0)==nbatches_break:
                    print 'Last', nbatches_break, 'batches had above 99.5% accuracy... stopping training'
                    break

        accuracy_per_batch.append(np.mean(acc)*100.0)

def eval(network,test_inputs,targets,cuda=False):
    network.eval()
    network.zero_grad()
    network.optimizer.zero_grad()
    # Define rnn private noise/spont_act
    spont_act = torch.randn(test_inputs.shape[0], test_inputs.shape[1], network.num_hidden, dtype=torch.float)/np.sqrt(network.num_hidden)
    ## Map inputs into RNN space, and add private noise
    rnn_input = network.w_in(test_inputs) + spont_act

    # Run RNN
    hidden, hn = network(rnn_input)
    
    # Compute outputs
    h2o = network.w_out(hidden) # Generate linear outputs
    outputs = network.sigmoid(h2o) # Pass through nonlinearity
    # Calculate loss
    M = targets*network.m_ij + 1
    loss = network.lossfunc(outputs,targets)
    loss = loss * M
    loss = torch.mean(loss)

    acc = [] # accuracy array
    for mb in range(targets.shape[1]):
        for out in range(targets.shape[2]):
            target_ind = targets[:,mb,out] == 1 
            if torch.sum(target_ind)==0: continue # If no responses for this finger continue
            target_ind = target_ind.byte() # identify response time points
            actual_response = outputs[:,mb,out] # Identify response time points
            response_periods = torch.nonzero(target_ind)
            thresh = .7 # decision thresh
            i = 0
            while i < len(response_periods): 
                tp = response_periods[i]
                target_resp = torch.ByteTensor([out]) # The correct target response
                response = torch.mean(actual_response[tp:tp+2])
                max_resp = torch.mean(outputs[tp:tp+2,mb,:],dim=0).argmax().byte()
                if max_resp==target_resp and response>thresh: # Make sure network response is correct respnose, and that it exceeds some threshold
                    acc.append(1.0)
                else:
                    acc.append(0)
                i += 2 # Each response period is 2 time points long, so go to next response period

    print '\tloss:', loss.item()
    print '\tAccuracy: ' + str(round(np.mean(acc)*100.0,4)) +'%'
    return outputs, hidden

def load_training_batches(cuda=False,filename='/projects3/SRActFlow/data/results/MODEL/TrialBatches_Default_WithDynamics2'):
    TrialObj = TrialBatches(filename=filename)
    inputs, outputs = TrialObj.loadTrainingBatches()

    inputs = inputs.float()
    outputs = outputs.float()
    if cuda==True:
        inputs = inputs.cuda()
        outputs = outputs.cuda()

    return inputs, outputs

def load_testset(cuda=False,filename='/projects3/SRActFlow/data/results/MODEL/TrialBatches_Default_WithDynamics2'):
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
                 NUM_BATCHES=150000,
                 NUM_TASKS_IN_TRAINSET=48,
                 NUM_TASK_TYPES_PER_BATCH=48,
                 NUM_TRIALS_PER_TASK=3,
                 NUM_TPs_PER_TASK=36,
                 NUM_INPUT_ELEMENTS=28,
                 NUM_OUTPUT_ELEMENTS=4,
                 filename='/projects3/SRActFlow/data/results/MODEL/TrialBatches_Default_WithDynamics2'):


        self.NUM_BATCHES = NUM_BATCHES
        self.NUM_TASKS_IN_TRAINSET = NUM_TASKS_IN_TRAINSET
        self.NUM_TASK_TYPES_PER_BATCH = NUM_TASK_TYPES_PER_BATCH
        self.NUM_TRIALS_PER_TASK = NUM_TRIALS_PER_TASK
        self.NUM_OUTPUT_ELEMENTS = NUM_OUTPUT_ELEMENTS
        self.NUM_TPs_PER_TASK = NUM_TPs_PER_TASK
        self.NUM_INPUT_ELEMENTS = NUM_INPUT_ELEMENTS
        self.splitTrainTestTaskSets(n_trainset=NUM_TASKS_IN_TRAINSET)
        self.filename = filename

    def createAllBatches(self,nproc=10):
        # Initialize empty tensor for batches
        #num_blocks = self.NUM_TASK_TYPES_PER_BATCH 
        num_blocks = self.NUM_TASK_TYPES_PER_BATCH 
        
        ### FIX TODO
        #batch_inputtensor = np.zeros((self.NUM_INPUT_ELEMENTS, self.NUM_TPs_PER_TASK, num_blocks, self.NUM_BATCHES))
        #batch_outputtensor = np.zeros((self.NUM_OUTPUT_ELEMENTS, self.NUM_TPs_PER_TASK, num_blocks, self.NUM_BATCHES))
        batch_inputtensor = np.zeros((self.NUM_INPUT_ELEMENTS, self.NUM_TPs_PER_TASK, len(self.trainRuleSet), self.NUM_BATCHES))
        batch_outputtensor = np.zeros((self.NUM_OUTPUT_ELEMENTS, self.NUM_TPs_PER_TASK, len(self.trainRuleSet), self.NUM_BATCHES))

        inputs = []
        for batch in range(self.NUM_BATCHES):
            shuffle = True
            inputs.append((self.trainRuleSet,self.NUM_TASK_TYPES_PER_BATCH,shuffle,batch))


        pool = mp.Pool(processes=nproc)
        results = pool.map_async(create_trial_batches,inputs).get()
        pool.close()
        pool.join()

        batch = 0
        for result in results:
            batch_inputtensor[:,:,:,batch] = result[0]
            batch_outputtensor[:,:,:,batch] = result[1]
            batch += 1

        # Construct test set
        nTasks = 64 - self.NUM_TASKS_IN_TRAINSET
        test_inputs, test_targets = create_trial_batches((self.testRuleSet,nTasks,shuffle,1))

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

        # Input dimensions: input features, nTPs, nMiniblocks, nBatches
        inputs = np.transpose(inputs, (3, 1, 2, 0)) # convert to: nBatches, nTPs, nMiniblocks, input dimensions
        outputs = np.transpose(outputs, (3, 1, 2, 0)) # convert to: nBatches, nTPs, nMiniblocks, input dimensions

        inputs = torch.from_numpy(inputs)
        outputs = torch.from_numpy(outputs)
        return inputs, outputs
    
    def loadTestset(self):
        h5f = h5py.File(self.filename + '.h5','r')
        inputs = h5f['test/inputs'][:].copy()
        outputs = h5f['test/outputs'][:].copy()
        h5f.close()

        # Input dimensions: input features, nTPs, nMiniblocks
        inputs = np.transpose(inputs, (1, 2, 0)) # convert to: nTPs, nMiniblocks, input dimensions
        outputs = np.transpose(outputs, (1, 2, 0)) # convert to: nTPs, nMiniblocks, input dimensions

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

def create_trial_batches((taskRuleSet,nTasks,shuffle,batchNum)):
    """
    Randomly generates a set of stimuli (nStimuli) for each task rule
    Will end up with 64 (task rules) * nStimuli total number of input stimuli
    
    If shuffle keyword is True, will randomly shuffle the training set
    Otherwise will start with taskrule1 (nStimuli), taskrule2 (nStimuli), etc.
    """
    if batchNum%100==0:
        print 'Running batch', batchNum
    
    stimuliSet = task.createSensoryInputs()

    # Create 1d array to randomly sample indices from
    stimIndices = np.arange(len(stimuliSet))
    taskIndices = np.arange(len(taskRuleSet))

    #nMiniblocks = nTasks
    nMiniblocks = len(taskIndices)
    nTasks = nTasks # Corresponds to the number of miniblocks
    n_trials_per_block = 3
    shuffle=True
    

    #randomTaskIndices = np.random.choice(taskIndices,len(taskIndices),replace=False)
    #randomTaskIndices = np.random.choice(taskIndices,nTasks,replace=False)
    #taskRuleSet2 = taskRuleSet.iloc[randomTaskIndices].copy(deep=True)
    #taskRuleSet = taskRuleSet.reset_index(drop=True)
    taskRuleSet = taskRuleSet.reset_index(drop=False)
    #taskRuleSet = taskRuleSet2.copy(deep=True)

    ####
    # Construct trial dynamics
    rule_ind = np.arange(12) # rules are the first 12 indices of input vector
    stim_ind = np.arange(12,28) # stimuli are the last 16 indices of input vector
    input_size = len(rule_ind) + len(stim_ind)
    n_tp_total = 36 # total length of miniblock -- cpro task details
    input_matrix = np.zeros((input_size,n_tp_total,nMiniblocks))
    output_matrix = np.zeros((4,n_tp_total,nMiniblocks))
    trial = 0
    for block in range(nMiniblocks):
        # Define trial dynamics
        n_tp_encoding = 5
        n_tp_encodingdelay = np.random.randint(2,9) # encoding delay is jittered from 2 - 8 trs
        n_tp_trial = 3
        n_tp_probedelay = 2
        n_tp_trial_end = n_tp_total - n_tp_encoding - n_tp_encodingdelay - n_tp_trial*3 - n_tp_probedelay*2 # full miniblock is 36 trs


        rand_stim_ind = np.random.choice(stimIndices,n_trials_per_block,replace=False)
        stimuliSet2 = stimuliSet.iloc[rand_stim_ind].copy(deep=True)
        stimuliSet2 = stimuliSet2.reset_index(drop=True)
        
        # Create trial array
        networkInputCode2 = []
        networkOutputCode2 = []
        tp = 0
        # Encoding
        for i in range(n_tp_encoding):
            input_matrix[rule_ind,tp,block] = taskRuleSet.Code[block] 
            tp += 1

        # Encoding delay
        tp += n_tp_encodingdelay

        # Trials
        for trial in range(n_trials_per_block):
            # First solve trial
            tmpresp, out_code = task.solveInputs(taskRuleSet.iloc[block], stimuliSet2.iloc[trial])

            # probe
            for i in range(n_tp_trial):
                input_matrix[stim_ind,tp,block] = stimuliSet2.Code[trial]
                # Commented out - response period will be limited to ITI following stimulus
                #output_matrix[:,tp,block] = out_code
                tp += 1

            # probe delay
            for i in range(n_tp_probedelay):
                output_matrix[:,tp,block] = out_code
                tp += 1
            
    if shuffle:
        ind = np.arange(input_matrix.shape[2],dtype=int)
        np.random.shuffle(ind)
        input_matrix = input_matrix[:,:,ind]
        output_matrix = output_matrix[:,:,ind]
        
    return input_matrix, output_matrix 

