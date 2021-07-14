import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
np.set_printoptions(suppress=True)


NUM_RULE_INPUTS = 64
NUM_SENSORY_INPUTS = 4
NUM_HIDDEN = 100
NUM_MOTOR_DECISION_OUTPUTS = 2

w_in = Variable(torch.Tensor(NUM_RULE_INPUTS + NUM_SENSORY_INPUTS, NUM_HIDDEN).uniform_(-0.5,0.5), requires_grad=True)
w_rec = Variable(torch.Tensor(NUM_HIDDEN, NUM_HIDDEN).uniform_(-0.5,0.5), requires_grad=True)
w_out = Variable(torch.Tensor(NUM_HIDDEN, NUM_MOTOR_DECISION_OUTPUTS).uniform_(-0.5,0.5), requires_grad=True)
bias = Variable(torch.Tensor(1, NUM_HIDDEN).uniform_(-0.5, 0), requires_grad=False)
drdt = 0.05


# Single input and output sequence.  You could have many of these.
inputs = [ [0, 1,   1, 1, -1, -1],
           [0, 0,   0, 0,  0,  0],
           [0, 0,   0, 0,  0,  0],
           [0, 0,   0, 0,  0,  0] ]

outputs = [ [0, 0],
            [0, 0],
            [0, 1],
            [0, 1] ]


learning_rate = 0.01
#for iteration_num in range(20000):
for iteration_num in range(100000):
    previous_r = Variable(torch.Tensor(1, NUM_HIDDEN).zero_(), requires_grad=False)
    error = 0
    for timestep in range(len(inputs)):
        u = Variable(torch.Tensor([inputs[timestep]]))
        target = Variable(torch.Tensor([outputs[timestep]]))

        # The neural network
        r = previous_r - drdt*previous_r + drdt* F.relu(previous_r.mm(w_rec) + u.mm(w_in) + bias)
        output = r.mm(w_out)

        error += torch.mean((output - target).pow(2))  # Mean squared error loss
        previous_r = r  # Recurrence

        if iteration_num % 1000 == 0:
            print(output.data.numpy())
    if iteration_num % 1000 == 0:
        print("Iteration ", iteration_num)
        print("loss: ", error.data[0])

    # Learning
    error.backward()
    w_in.data -= learning_rate*w_in.grad.data; w_in.grad.data.zero_()
    w_rec.data -= learning_rate*w_rec.grad.data; w_rec.grad.data.zero_()
    w_out.data -= learning_rate*w_out.grad.data; w_out.grad.data.zero_()



