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
                 NUM_RULE_INPUTS=12,
                 NUM_SENSORY_INPUTS=16,
                 NUM_HIDDEN=128,
                 NUM_MOTOR_DECISION_OUTPUTS=4,
                 cuda=False):

        self.NUM_RULE_INPUTS = NUM_RULE_INPUTS
        self.NUM_SENSORY_INPUTS =  NUM_SENSORY_INPUTS
        self.NUM_HIDDEN = NUM_HIDDEN
        self.NUM_MOTOR_DECISION_OUTPUTS = NUM_MOTOR_DECISION_OUTPUTS
        self.cuda = cuda

        super(RNN,self).__init__(input_size=self.NUM_HIDDEN,
                                 hidden_size=self.NUM_HIDDEN,
                                 num_layers=1,
                                 nonlinearity='relu',
                                 bias=True,
                                 batch_first=False)

        #self.splitTrainTestTaskSets(n_trainset=48)
        
#
        self.w_in = torch.nn.Linear(NUM_SENSORY_INPUTS+NUM_RULE_INPUTS,NUM_HIDDEN)
#        self.w_rec = torch.nn.Linear(NUM_HIDDEN,NUM_HIDDEN)
        self.w_out = torch.nn.Linear(NUM_HIDDEN,NUM_MOTOR_DECISION_OUTPUTS)
#        self.func = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

        # Construct optimizer
        learning_rate = .0001
        ##optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def train(self, inputs, targets):
        """Train network"""
        lossfunc = torch.nn.MSELoss()

        self.zero_grad()
        self.optimizer.zero_grad()
        nTPs = inputs.shape[0]
        nMiniblocks = inputs.shape[1]

#        # Define noise 
#        input_noise = torch.randn(inputs.shape, dtype=torch.float)/np.sqrt((self.NUM_SENSORY_INPUTS+self.NUM_RULE_INPUTS))
#        #spont_act = torch.randn(nMiniblocks, nTPs, self.NUM_HIDDEN, dtype=torch.float)/np.sqrt(self.NUM_HIDDEN)
#        spont_act = torch.randn(1, nMiniblocks, self.NUM_HIDDEN, dtype=torch.float)/np.sqrt(self.NUM_HIDDEN)
#
#
#        # Run RNN
#        outputs, hidden = self(inputs+input_noise, spont_act)
#        #outputs, hidden = self(inputs+input_noise)
#        # Compute outputs
#        h2o = self.w_out(outputs) # Generate linear outupts
#        outputs = self.sigmoid(h2o) # Pass through nonlinearity


        # Define rnn private noise/spont_act
        spont_act = torch.randn(nTPs, nMiniblocks, self.NUM_HIDDEN, dtype=torch.float)/np.sqrt(self.NUM_HIDDEN)
        ## Map inputs into RNN space, and add private noise
        rnn_input = self.w_in(inputs) + spont_act

        outputs, hidden = self(rnn_input)
        
        #input_noise = torch.randn(nTPs, nMiniblocks, self.NUM_SENSORY_INPUTS+self.NUM_RULE_INPUTS, dtype=torch.float)/np.sqrt(self.NUM_SENSORY_INPUTS+self.NUM_RULE_INPUTS)
        #outputs, hidden = self(inputs+input_noise)


        # Compute outputs
        h2o = self.w_out(outputs) # Generate linear outupts
        outputs = self.sigmoid(h2o) # Pass through nonlinearity

        # Calculate loss
        M = targets*4
        M = M + 1
        lossfunc = torch.nn.MSELoss(reduce=False)
        loss = lossfunc(outputs,targets)
        loss = loss * M
        loss = torch.mean(loss)
        
        loss.backward()
        self.optimizer.step() # Update parameters using optimizer

        return outputs, targets, loss

    def runTrainingEpoch(self,train_inputs,train_outputs,
                         NUM_TRAINING_ITERATIONS=100000,
                         NUM_TRAINING_RULES_PER_EPOCH=3,
                         NUM_TRAINING_STIMULI_PER_RULE=100,
                         cuda=False):

        accuracyPerEpoch = []

        batchTrainingArray = np.arange(NUM_TRAINING_ITERATIONS)

        np.random.shuffle(batchTrainingArray)

        timestart = time.time()
        for batch in np.arange(NUM_TRAINING_ITERATIONS):
            batchID = batchTrainingArray[batch]

            if cuda:
                train_input_batch = train_inputs[batchID,:,:,:].cuda()
                train_output_batch = train_outputs[batchID,:,:,:].cuda()
            else:
                train_input_batch = train_inputs[batchID,:,:,:]
                train_output_batch = train_outputs[batchID,:,:,:]

            outputs, targets, loss = self.train(train_inputs[batchID,:,:,:],
                                                train_outputs[batchID,:,:,:])

            if batch % 500 == 0:
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

                timeend = time.time()
                print 'Iteration:', batch
                print '\tloss:', loss.item()
                print 'Time elapsed...', timeend-timestart
                timestart = timeend
                print '\tAccuracy: ' + str(round(np.mean(acc)*100.0,4)) +'%'

            accuracyPerEpoch.append(np.mean(acc)*100.0)
            if batch>10:
                if np.sum(np.asarray(accuracyPerEpoch[-3:])>95.0)==3:
                    print 'Last 3000 batches had above 95% accuracy... stopping training'
                    break


    def splitTrainTestTaskSets(self,n_trainset=32):
        self.n_trainset = n_trainset
        self.n_testset = 64 - n_trainset

        taskRuleSet = task.createRulePermutations()
        trainRuleSet, testRuleSet = task.createTrainTestTaskRules(taskRuleSet,nTrainSet=n_trainset, nTestSet=self.n_testset)

        self.taskRuleSet = taskRuleSet
        self.trainRuleSet = trainRuleSet
        self.testRuleSet = testRuleSet

    def loadTrainingBatch(self,cuda=False,filename='/projects3/SRActFlow/data/results/MODEL/TrialBatches_Default_WithDynamics'):
        TrialObj = TrialBatches(filename=filename)
        inputs, outputs = TrialObj.loadTrainingBatches()

        self.nTPs = inputs.shape[1]
        self.nMiniblocks = inputs.shape[2]

        inputs = inputs.float()
        outputs = outputs.float()
        if cuda==True:
            inputs = inputs.cuda()
            outputs = outputs.cuda()

        return inputs, outputs

    def loadTestset(self,cuda=False,filename='/projects3/SRActFlow/data/results/MODEL/TrialBatches_Default_WithDynamics'):
        TrialObj = TrialBatches(filename=filename)
        test_inputs, test_outputs = TrialObj.loadTestset()

        self.nTPs = test_inputs.shape[1]
        self.nMiniblocks = test_inputs.shape[2]

        test_inputs = test_inputs.float()
        test_outputs = test_outputs.float()
        if cuda==True:
            test_inputs = test_inputs.cuda()
            test_outputs = test_outputs.cuda()
        return test_inputs, test_outputs

    def evaluateTestset(self,test_inputs,targets,filename='/projects3/SRActFlow/data/results/MODEL/TrialBatches_Default_WithDynamics',cuda=False):
        #self.loadTestset(cuda=cuda,filename=filename)
        #inputs = inputs.t()
        #targets = targets.t()

        self.zero_grad()
        self.optimizer.zero_grad()
        # Define rnn private noise/spont_act
        spont_act = torch.randn(test_inputs.shape[0], test_inputs.shape[1], self.NUM_HIDDEN, dtype=torch.float)/np.sqrt(self.NUM_HIDDEN)
        ## Map inputs into RNN space, and add private noise
        rnn_input = self.w_in(test_inputs) + spont_act

        # Run RNN
        outputs, hidden = self(rnn_input)
        
        # Compute outputs
        h2o = self.w_out(outputs) # Generate linear outputs
        outputs = self.sigmoid(h2o) # Pass through nonlinearity
        # Calculate loss
        M = targets*4
        M = M + 1
        lossfunc = torch.nn.MSELoss(reduce=False)
        loss = lossfunc(outputs,targets)
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

class TrialBatches(object):
    """
    Batch trials
    """
    def __init__(self,
                 NUM_BATCHES=100000,
                 NUM_TASKS_IN_TRAINSET=48,
                 NUM_TASK_TYPES_PER_BATCH=32,
                 NUM_TRIALS_PER_TASK=3,
                 NUM_TPs_PER_TASK=36,
                 NUM_INPUT_ELEMENTS=28,
                 NUM_OUTPUT_ELEMENTS=4,
                 filename='/projects3/SRActFlow/data/results/MODEL/TrialBatches_Default_WithDynamics'):


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
        num_blocks = self.NUM_TASK_TYPES_PER_BATCH 
        
        batch_inputtensor = np.zeros((self.NUM_INPUT_ELEMENTS, self.NUM_TPs_PER_TASK, num_blocks, self.NUM_BATCHES))
        batch_outputtensor = np.zeros((self.NUM_OUTPUT_ELEMENTS, self.NUM_TPs_PER_TASK, num_blocks, self.NUM_BATCHES))

        #for batch in range(self.NUM_BATCHES):
        #    if batch%100==0:
        #        print 'Creating batch', batch, '/', self.NUM_BATCHES

        #    df, inputs, outputs = self.createTrainingBatch(nStimuli=self.NUM_TRIALS_PER_TASK,
        #                                                   nTasks=self.NUM_TASK_TYPES_PER_BATCH,delay=False) # 64 * 20 stimuli
        inputs = []
        for batch in range(self.NUM_BATCHES):
            shuffle = True
            inputs.append((self.trainRuleSet,self.NUM_TASK_TYPES_PER_BATCH,shuffle,batch))

        pool = mp.Pool(processes=nproc)
        results = pool.map_async(createTrialBatchesWithDynamics,inputs).get()
        pool.close()
        pool.join()

        batch = 0
        for result in results:
            batch_inputtensor[:,:,:,batch] = result[0]
            batch_outputtensor[:,:,:,batch] = result[1]
            batch += 1

        # Construct test set
        nTasks = 64 - self.NUM_TASKS_IN_TRAINSET
        test_inputs, test_targets = createTrialBatchesWithDynamics((self.testRuleSet,nTasks,shuffle,1))

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

def createTrialBatchesWithDynamics((taskRuleSet,nTasks,shuffle,batchNum)):
    """
    Randomly generates a set of stimuli (nStimuli) for each task rule
    Will end up with 64 (task rules) * nStimuli total number of input stimuli
    
    If shuffle keyword is True, will randomly shuffle the training set
    Otherwise will start with taskrule1 (nStimuli), taskrule2 (nStimuli), etc.
    """
    if batchNum%100==0:
        print 'Running batch', batchNum

    nMiniblocks = nTasks
    nTasks = nTasks # Corresponds to the number of miniblocks
    n_trials_per_block = 3
    shuffle=True
    
    stimuliSet = task.createSensoryInputs()

    networkIO_DataFrame = {}
    networkIO_DataFrame['LogicRule'] = []
    networkIO_DataFrame['SensoryRule'] = []
    networkIO_DataFrame['MotorRule'] = []
    networkIO_DataFrame['Color1'] = []
    networkIO_DataFrame['Color2'] = []
    networkIO_DataFrame['Orientation1'] = []
    networkIO_DataFrame['Orientation2'] = []
    networkIO_DataFrame['Pitch1'] = []
    networkIO_DataFrame['Pitch2'] = []
    networkIO_DataFrame['Constant1'] = []
    networkIO_DataFrame['Constant2'] = []
    networkIO_DataFrame['MotorResponse'] = []

    # Create 1d array to randomly sample indices from
    stimIndices = np.arange(len(stimuliSet))
    taskIndices = np.arange(len(taskRuleSet))
    
    randomTaskIndices = np.random.choice(taskIndices,nTasks,replace=False)
    taskRuleSet2 = taskRuleSet.iloc[randomTaskIndices].copy(deep=True)
    taskRuleSet2 = taskRuleSet2.reset_index(drop=True)
    taskRuleSet = taskRuleSet2.copy(deep=True)

#    networkInputCode = []
#    networkOutputCode = []
#    for taskrule in taskRuleSet.index:
#        
#        randomStimuliIndices = np.random.choice(stimIndices,n_trials_per_block,replace=False)
#        stimuliSet2 = stimuliSet.iloc[randomStimuliIndices].copy(deep=True)
#        stimuliSet2 = stimuliSet2.reset_index(drop=True)
#        
#        for trial in stimuliSet2.index:
#
#            networkInputCode.append(np.hstack((taskRuleSet.Code[taskrule], stimuliSet2.Code[trial])))
#            tmpresp, tmpcode = task.solveInputs(taskRuleSet.iloc[taskrule], stimuliSet2.iloc[trial])
#            networkOutputCode.append(tmpcode)

    

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
