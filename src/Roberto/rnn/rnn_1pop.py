import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init, functional as F

class CTRNN(nn.Module):  # to create a recurrent neural network with an input layer (does not have an output layer yet)
    def __init__(self, input_size, hidden_size, **kwargs):
        # define all the network params
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = "cpu"  # ("mps" if torch.backends.mps.is_available() else "cpu") # define device as machine gpu
        self.tau = kwargs.get('tau', 50)  # time constant of the network
        self.bias = kwargs.get('bias', False)  # bias weights to be set to 0 as default or 'False'
        self.dt = kwargs.get('dt', 1)  # time step for the continuous time recurrent neural network
        self.train_initial_state = kwargs.get('train_initial_state',
                                              True)  # whether to train the initial condition of the network or not
        self.alpha = self.dt / self.tau  # in most cases dt = 1ms and tau = 50.
        # default definition of input and recurrent weights
        self.input2h = nn.Linear(input_size, hidden_size, bias=self.bias).to(self.device)
        self.h2h = nn.Linear(hidden_size, hidden_size, bias=self.bias).to(self.device)

        # initialize the input and hidden weights
        init.normal_(self.input2h.weight, mean=0, std=0.5)
        init.normal_(self.h2h.weight, mean=0, std=1 / np.sqrt(
            hidden_size))  # initialize recurrent weights to g/sqrt(N) (with spectral radius of 1)

        if self.train_initial_state:  # if we want to train initial conditions of the network
            # initial_hidden_tensor = torch.zeros(1, hidden_size, requires_grad=True,device=self.device) # if you want to initialize the network activity to 0s
            initial_hidden_tensor = torch.rand(1, self.hidden_size, requires_grad=self.train_initial_state,
                                               device=self.device) * 2 - 1
            self.initial_hidden = nn.Parameter(initial_hidden_tensor)
            self.register_parameter('initial_hidden', self.initial_hidden)
        else:
            initial_hidden_tensor = torch.zeros(1, hidden_size, requires_grad=False)
            self.register_buffer('initial_hidden',
                                 initial_hidden_tensor)  # Using register_buffer for non-trainable tensors
        self.to(self.device)

    def init_hidden(self, batch_size):
        # return torch.zeros(batch_size, self.hidden_size, device=self.device)
        return self.initial_hidden.repeat(batch_size, 1)

    def recurrence(self, input, hidden):  # define dynamics of the rnn
        h_new = torch.sigmoid(self.input2h(input) + self.h2h(hidden))
        h_new = hidden * (1 - self.alpha) + h_new * self.alpha
        h_new = h_new.to(self.device)
        return h_new

    def forward(self, input, hidden=None):
        input = input.to(self.device)
        if hidden is None:
            hidden = self.init_hidden(input.shape[1])  # Use the learnable initial hidden state
        output = []
        for i in range(input.size(0)):
            hidden = self.recurrence(input[i], hidden)
            output.append(hidden)
        output = torch.stack(output, dim=0)
        return output, hidden


class RNNNet(nn.Module):  # to create the full model with the recurrent part above and an output layer
    def __init__(self, input_size, hidden_size, output_size, **kwargs):
        super().__init__()
        self.bias = kwargs.get('bias', False)
        self.device = "cpu"  # ("mps" if torch.backends.mps.is_available() else "cpu")
        self.rnn = CTRNN(input_size, hidden_size, **kwargs).to(self.device)  # continuous time RNN
        self.fc = nn.Linear(hidden_size, output_size, bias=self.bias).to(self.device)  # add linear readout layer
        init.normal_(self.fc.weight, mean=0, std=0.5)

    def forward(self, x, hidden=None):
        x = x.to(self.device)  # input tensor of shape (Seq Len, Batch, Input Dim)
        rnn_output, _ = self.rnn(x, hidden=hidden)  # get activity of the rnn
        out = torch.stack((rnn_output.mean(axis=-1), rnn_output.std(axis=-1) ** 2), dim=-1)  # self.fc(rnn_output)  # linear output to the activity of the rnn
        return out, rnn_output