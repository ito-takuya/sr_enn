# Taku Ito
# 03/11/2019
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




class RNN(torch.nn.Module):
    """
    Neural network object
    """
    def __init__(self,
                 NUM_RULE_INPUTS=12,
                 NUM_SENSORY_INPUTS=16,
                 NUM_HIDDEN=128,
                 NUM_MOTOR_DECISION_OUTPUTS=4):

        super(RNN,self).__init__()

        self.NUM_RULE_INPUTS = NUM_RULE_INPUTS
        self.NUM_SENSORY_INPUTS =  NUM_SENSORY_INPUTS
        self.NUM_HIDDEN = NUM_HIDDEN
        self.NUM_MOTOR_DECISION_OUTPUTS = NUM_MOTOR_DECISION_OUTPUTS
        #self.splitTrainTestTaskSets(n_trainset=48)
        
        # Initialize RNN units
        self.units = torch.nn.Parameter(self.initHidden())

        #self.w_in_stim = Variable(torch.Tensor(NUM_SENSORY_INPUTS, NUM_HIDDEN).uniform_(-0.5,0.5), requires_grad=True)
        #self.w_in_rules = Variable(torch.Tensor(NUM_RULE_INPUTS, NUM_HIDDEN).uniform_(-0.5,0.5), requires_grad=True)
        #self.w_in = Variable(torch.Tensor(NUM_RULE_INPUTS+NUM_SENSORY_INPUTS, NUM_HIDDEN).uniform_(-0.5,0.5), requires_grad=True)
        #self.w_rec = Variable(torch.Tensor(NUM_HIDDEN, NUM_HIDDEN).uniform_(-0.5,0.5), requires_grad=True)
        #self.w_out = Variable(torch.Tensor(NUM_HIDDEN, NUM_MOTOR_DECISION_OUTPUTS).uniform_(-0.5,0.5), requires_grad=True)
        #bias = Variable(torch.Tensor(1, NUM_HIDDEN).uniform_(-1, 0), requires_grad=False)
        #self.bias = Variable(torch.Tensor(1, NUM_HIDDEN).uniform_(-1, 0), requires_grad=True)


        self.w_in = torch.nn.Linear(NUM_SENSORY_INPUTS+NUM_RULE_INPUTS,NUM_HIDDEN)
        self.w_rec = torch.nn.Linear(NUM_HIDDEN,NUM_HIDDEN)
        self.w_out = torch.nn.Linear(NUM_HIDDEN,NUM_MOTOR_DECISION_OUTPUTS)
        self.func = torch.nn.ReLU()


        # Construct optimizer
        #learning_rate = 1e-4
        learning_rate = .001
        #learning_rate = .01
        #learning_rate = .05
        ##learning_rate = 0.01 # orig
        ##optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def initHidden(self):
        return torch.zeros(1, self.NUM_HIDDEN)

    def forward(self,input,hidden,drdt=0.05):
        """Go to next timepoint"""
        previous_r = self.units
        i2h = self.w_in(input) # compute input to recurrent units
        h2h = self.w_rec(previous_r) # compute recurrence
        r = previous_r - drdt*previous_r + drdt*self.func(i2h + h2h) # compute next time point

        # Compute outputs
        h2o = self.w_out(r) # Generate linear outupts
        output = self.func(h2o) # Pass through nonlinearity

        # update units
        #self.units = torch.nn.Parameter(r)

        return output, hidden

    def train(self, inputs, targets):
        """Train network"""
        lossfunc = torch.nn.MSELoss()

        hidden = self.units 

        self.zero_grad()
        self.optimizer.zero_grad()
        #learning_rate = 1e-4
        #learning_rate = .001
        ##learning_rate = 0.01 # orig
        ##optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        #optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)

#        outputs = torch.FloatTensor(size=(inputs.size()[0],4))
#        for i in range(inputs.size()[0]):
#            outputs[i,:], hidden = self.forward(inputs[i],hidden)
        outputs, hidden = self.forward(inputs,hidden)


        # Compute gradient of loss
        loss = lossfunc(outputs,targets)
        print outputs.shape
        print targets.shape
        print loss
        loss.backward()

        self.optimizer.step() # Update parameters using optimizer

        return outputs, targets, loss

    def loadTrainingBatch(self,cuda=False,filename='/projects3/SRActFlow/data/results/MODEL/TrialBatches_Default'):
        TrialObj = TrialBatches(filename=filename)
        inputs, outputs = TrialObj.loadTrainingBatches()
        inputs = inputs.float()
        outputs = outputs.float()
        if cuda==True:
            inputs = inputs.cuda()
            outputs = outputs.cuda()
        self.input_tensors_allbatches = inputs
        self.output_tensors_allbatches = outputs

    def loadTestset(self,cuda=False,filename='/projects3/SRActFlow/data/results/MODEL/TrialBatches_Default'):
        TrialObj = TrialBatches(filename=filename)
        inputs, outputs = TrialObj.loadTestset()
        inputs = inputs.float()
        outputs = outputs.float()
        if cuda==True:
            inputs = inputs.cuda()
            outputs = outputs.cuda()
        self.test_inputs = inputs
        self.test_targets = outputs

    def runTrainingEpoch(self,
                         NUM_TRAINING_ITERATIONS=100000,
                         NUM_TRAINING_RULES_PER_EPOCH=3,
                         NUM_TRAINING_STIMULI_PER_RULE=100,
                         cuda=False):

        accuracyPerEpoch = []

        batchTrainingArray = np.arange(NUM_TRAINING_ITERATIONS)

        np.random.shuffle(batchTrainingArray)

        for batch in np.arange(NUM_TRAINING_ITERATIONS):
            batchID = batchTrainingArray[batch]
            # No trial dynamics so forget previous input

            # Increase number of presented tasks with number of increased iterations
            # Don't allow more than 10 task rules per epoch, since it will just slow training down
            #if batch % 2000 == 0:
            #    if NUM_TRAINING_RULES_PER_EPOCH < 10:
            #        NUM_TRAINING_RULES_PER_EPOCH += 1

            #df, inputs, outputs = self.createTrainingBatch(nStimuli=NUM_TRAINING_STIMULI_PER_RULE,
            #                                               nTasks=NUM_TRAINING_RULES_PER_EPOCH,delay=False) # 64 * 20 stimuli
            #inputs = torch.from_numpy(inputs)
            #outputs = torch.from_numpy(outputs)

            #inputs = inputs.float()
            #outputs = outputs.float()
            #if cuda==True:
            #    inputs = inputs.cuda()
            #    outputs = outputs.cuda()
            outputs, targets, loss = self.train(self.input_tensors_allbatches[:,:,batchID].t(),
                                                self.output_tensors_allbatches[:,:,batchID].t())

            if batch % 5000 == 0:
                acc = []
                targets = targets.cpu()
                outputs = outputs.cpu()
                for timestep in range(targets.size()[0]):
                    # Calculate accuracy:
                    if np.sum(np.asarray(targets.data[timestep,:]))!=0:
                        distance = np.abs(1.0-outputs.data[timestep,:])
                        #print np.where(distance == distance.min())[0][0]
                        #print np.where(np.asarray(targets.data[timestep,:]))[0][0]
                        if np.where(distance == distance.min())[0][0] == np.where(np.asarray(targets.data[timestep,:]))[0][0]:
                            acc.append(1.0)
                        else:
                            acc.append(0.0)
                print 'Iteration:', batch
                print '\tloss:', loss.item()
                print '\tAccuracy: ' + str(round(np.mean(acc)*100.0,4)) +'%'

            accuracyPerEpoch.append(np.mean(acc)*100.0)
            if batch>10:
                if np.sum(np.asarray(accuracyPerEpoch[-10:])>99.0)==10:
                    print 'Last 10 epochs had above 99% accuracy... stopping training'
                    break


    def splitTrainTestTaskSets(self,n_trainset=32):
        self.n_trainset = n_trainset
        self.n_testset = 64 - n_trainset

        taskRuleSet = task.createRulePermutations()
        trainRuleSet, testRuleSet = task.createTrainTestTaskRules(taskRuleSet,nTrainSet=n_trainset, nTestSet=self.n_testset)

        self.taskRuleSet = taskRuleSet
        self.trainRuleSet = trainRuleSet
        self.testRuleSet = testRuleSet

    def evaluateTestset(self,filename='/projects3/SRActFlow/data/results/MODEL/TrialBatches_Default',cuda=False):

        self.loadTestset(cuda=cuda,filename=filename)
        #inputs = inputs.t()
        #targets = targets.t()

        hidden = self.units 

        self.zero_grad()
        self.optimizer.zero_grad()
        outputs, hidden = self.forward(self.test_inputs,hidden)

        acc = []
        for timestep in range(self.test_targets.size()[0]):
            # Calculate accuracy:
            if np.sum(np.asarray(self.test_targets.data[timestep,:]))!=0:
                distance = np.abs(1.0-outputs.data[timestep,:])
                #print np.where(distance == distance.min())[0][0]
                #print np.where(np.asarray(targets.data[timestep,:]))[0][0]
                if np.where(distance == distance.min())[0][0] == np.where(np.asarray(self.test_targets.data[timestep,:]))[0][0]:
                    acc.append(1.0)
                else:
                    acc.append(0.0)
        print '\tAccuracy: ' + str(round(np.mean(acc)*100.0,4)) +'%'





class TrialBatches(object):
    """
    Batch trials
    """
    def __init__(self,
                 NUM_BATCHES=100000,
                 NUM_TASKS_IN_TRAINSET=48,
                 NUM_TASK_TYPES_PER_BATCH=32,
                 NUM_TRIALS_PER_TASK=5,
                 NUM_INPUT_ELEMENTS=28,
                 NUM_OUTPUT_ELEMENTS=4,
                 filename='/projects3/SRActFlow/data/results/MODEL/TrialBatches_Default'):


        self.NUM_BATCHES = NUM_BATCHES
        self.NUM_TASKS_IN_TRAINSET = NUM_TASKS_IN_TRAINSET
        self.NUM_TASK_TYPES_PER_BATCH = NUM_TASK_TYPES_PER_BATCH
        self.NUM_TRIALS_PER_TASK = NUM_TRIALS_PER_TASK
        self.NUM_OUTPUT_ELEMENTS = NUM_OUTPUT_ELEMENTS
        self.NUM_INPUT_ELEMENTS = NUM_INPUT_ELEMENTS
        self.splitTrainTestTaskSets(n_trainset=NUM_TASKS_IN_TRAINSET)
        self.filename = filename

    def createAllBatches(self,nproc=10):

        # Initialize empty tensor for batches
        num_trials = self.NUM_TASK_TYPES_PER_BATCH * self.NUM_TRIALS_PER_TASK

        batch_inputtensor = np.zeros((self.NUM_INPUT_ELEMENTS, num_trials, self.NUM_BATCHES))
        batch_outputtensor = np.zeros((self.NUM_OUTPUT_ELEMENTS, num_trials, self.NUM_BATCHES))

        #for batch in range(self.NUM_BATCHES):
        #    if batch%100==0:
        #        print 'Creating batch', batch, '/', self.NUM_BATCHES

        #    df, inputs, outputs = self.createTrainingBatch(nStimuli=self.NUM_TRIALS_PER_TASK,
        #                                                   nTasks=self.NUM_TASK_TYPES_PER_BATCH,delay=False) # 64 * 20 stimuli
        inputs = []
        for batch in range(self.NUM_BATCHES):
            delay = False
            shuffle = True
            inputs.append((self.trainRuleSet,self.NUM_TASK_TYPES_PER_BATCH,self.NUM_TRIALS_PER_TASK,delay,shuffle,batch))
                    
        pool = mp.Pool(processes=nproc)
        results = pool.map_async(createTrialBatches,inputs).get()
        pool.close()
        pool.join()

        batch = 0
        for result in results:
            batch_inputtensor[:,:,batch] = result[0].T
            batch_outputtensor[:,:,batch] = result[1].T
            batch += 1

        # Construct test set
        n_trials_per_task = 200
        nTasks = 64 - self.NUM_TASKS_IN_TRAINSET
        test_inputs, test_targets = createTrialBatches((self.testRuleSet,nTasks,n_trials_per_task,delay,shuffle,1))

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

        inputs = torch.from_numpy(inputs)
        outputs = torch.from_numpy(outputs)
        return inputs, outputs
    
    def loadTestset(self):

        h5f = h5py.File(self.filename + '.h5','r')
        inputs = h5f['test/inputs'][:].copy()
        outputs = h5f['test/outputs'][:].copy()
        h5f.close()

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

def createTrialBatches((taskRuleSet,nTasks,nTrialsPerTask,delay,shuffle,batchNum)):
    """
    Randomly generates a set of stimuli (nStimuli) for each task rule
    Will end up with 64 (task rules) * nStimuli total number of input stimuli
    
    If shuffle keyword is True, will randomly shuffle the training set
    Otherwise will start with taskrule1 (nStimuli), taskrule2 (nStimuli), etc.
    """
    if batchNum%100==0:
        print 'Running batch', batchNum

    nStimuli = nTrialsPerTask
    nTasks = nTasks
    delay=False
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

    networkInputCode = []
    networkOutputCode = []
    for taskrule in taskRuleSet.index:
        
        randomStimuliIndices = np.random.choice(stimIndices,nStimuli,replace=False)
        stimuliSet2 = stimuliSet.iloc[randomStimuliIndices].copy(deep=True)
        stimuliSet2 = stimuliSet2.reset_index(drop=True)
        
        for stim in stimuliSet2.index:

            networkInputCode.append(np.hstack((taskRuleSet.Code[taskrule], stimuliSet2.Code[stim])))
            tmpresp, tmpcode = task.solveInputs(taskRuleSet.iloc[taskrule], stimuliSet2.iloc[stim])
            networkOutputCode.append(tmpcode)

            # Task rule info
            networkIO_DataFrame['LogicRule'].append(taskRuleSet.Logic[taskrule])
            networkIO_DataFrame['SensoryRule'].append(taskRuleSet.Sensory[taskrule])
            networkIO_DataFrame['MotorRule'].append(taskRuleSet.Motor[taskrule])
            # Stimuli info
            networkIO_DataFrame['Color1'].append(stimuliSet2.Color1[stim])
            networkIO_DataFrame['Color2'].append(stimuliSet2.Color2[stim])
            networkIO_DataFrame['Orientation1'].append(stimuliSet2.Orientation1[stim])
            networkIO_DataFrame['Orientation2'].append(stimuliSet2.Orientation2[stim])
            networkIO_DataFrame['Pitch1'].append(stimuliSet2.Pitch1[stim])
            networkIO_DataFrame['Pitch2'].append(stimuliSet2.Pitch2[stim])
            networkIO_DataFrame['Constant1'].append(stimuliSet2.Constant1[stim])
            networkIO_DataFrame['Constant2'].append(stimuliSet2.Constant2[stim])
            # Motor info
            networkIO_DataFrame['MotorResponse'].append(tmpresp)
            

    tmpdf = pd.DataFrame(networkIO_DataFrame)
    
    if shuffle:
        ind = np.arange(len(tmpdf),dtype=int)
        np.random.shuffle(ind)
        networkIO_DataFrame = tmpdf.iloc[ind]
        networkInputCode = np.asarray(networkInputCode)[ind]
        networkOutputCode = np.asarray(networkOutputCode)[ind]

    # Add delay (i.e., 0 inputs & 0 outputs just incase)
    if delay:
        networkInputCode2 = []
        networkOutputCode2 = []
        nDelays = 1
        
        for index in range(len(networkIO_DataFrame)):
            networkInputCode2.append(networkInputCode[index])
            networkOutputCode2.append(networkOutputCode[index])
            
            for delay in range(nDelays):
                networkInputCode2.append(np.zeros((len(networkInputCode[index]),)))
                networkOutputCode2.append(np.zeros((len(networkOutputCode[index]),)))
            
            
        networkInputCode = np.asarray(networkInputCode2)
        networkOutputCode = np.asarray(networkOutputCode2)
            
        
        
    return networkInputCode, networkOutputCode

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

    
#    if shuffle:
#        ind = np.arange(len(tmpdf),dtype=int)
#        np.random.shuffle(ind)
#        networkIO_DataFrame = tmpdf.iloc[ind]
#        networkInputCode = np.asarray(networkInputCode)[ind]
#        networkOutputCode = np.asarray(networkOutputCode)[ind]

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
                output_matrix[:,tp,block] = out_code
                tp += 1
            
            # If trial isn't last trial, then include ITI
            if trial!=2:
                tp += n_tp_probedelay
        
    return input_matrix, output_matrix 
