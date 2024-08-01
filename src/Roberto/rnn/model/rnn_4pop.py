import time
import random
import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from torch.nn import init, functional as F


class MaskedExpLinear(nn.Linear):
    def __init__(self, in_features, out_features, mask: torch.tensor, bias=True):
        super().__init__(in_features, out_features, bias)
        self.mask = mask

    def forward(self, input):
        return F.linear(input, torch.relu(self.weight) @ self.mask, self.bias)

class MaskedInputLinear(nn.Linear):
    def __init__(self, in_features, out_features, mask: torch.tensor, bias=True):
        super().__init__(in_features, out_features, bias)
        self.mask_input = mask

    def forward(self, input):
        return F.linear(input, torch.abs(self.weight) * self.mask_input, self.bias)


class CTRNN4pop(nn.Module):  # to create a recurrent neural network with an input layer (does not have an output layer yet)
    def __init__(self, input_size, hidden_size, **kwargs):
        # define all the network params
        super().__init__()
        self.percentage_excitatory = 0.25
        self.number_neurons_excitatory = int(self.percentage_excitatory * hidden_size)
        self.mask = torch.diag(torch.ones(hidden_size))
        self.mask[self.number_neurons_excitatory:, self.number_neurons_excitatory:] *= -1
        self.mask_input = torch.ones((hidden_size, input_size))
        self.mask_input[2*self.number_neurons_excitatory:, :] *= 0

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
        self.input2h = MaskedInputLinear(input_size, hidden_size, self.mask_input, bias=False).to(self.device)
        self.h2h = MaskedExpLinear(hidden_size, hidden_size, mask=self.mask, bias=self.bias).to(self.device)

        # initialize the input and hidden weights
        init.trunc_normal_(self.input2h.weight, mean=0, std=0.5, a=0)
        # init.trunc_normal_(self.h2h.weight, mean=0, std=1 / np.sqrt(hidden_size), a=0)  # initialize recurrent weights to g/sqrt(N) (with spectral radius of 1)
        # self.h2h.weight = torch.nn.Parameter(torch.cat((torch.from_numpy(np.random.exponential(1, (hidden_size, self.number_neurons_excitatory)).astype(np.float32)),
        #                                                        torch.from_numpy(np.random.exponential(0.2, (hidden_size, hidden_size-self.number_neurons_excitatory)).astype(np.float32))), dim=-1))
        mu_small = 0
        mu_large = 5
        a_small, b_small = (0 - mu_small) / 0.5, (2 - mu_small) / 0.5
        a_large, b_large = (0 - mu_large) / 0.5, (2 - mu_large) / 0.5
        truncnorm_small = stats.truncnorm(a_small, b_small, loc=mu_small, scale=0.5)
        truncnorm_large = stats.truncnorm(a_large, b_large, loc=mu_small, scale=0.5)
        self.h2h.weight = torch.nn.Parameter(torch.cat((torch.from_numpy(truncnorm_large.rvs(size=(hidden_size, self.number_neurons_excitatory)).astype(np.float32)),
                                                        torch.from_numpy(truncnorm_small.rvs(size=(hidden_size, hidden_size - self.number_neurons_excitatory)).astype(np.float32))), dim=-1))


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


class RNNNet4pop(nn.Module):  # to create the full model with the recurrent part above and an output layer
    def __init__(self, input_size, hidden_size, output_size, **kwargs):
        super().__init__()
        self.percentage_excitatory = 0.25
        self.number_neurons_excitatory = int(self.percentage_excitatory * hidden_size)
        self.bias = kwargs.get('bias', False)
        self.device = "cpu"  # ("mps" if torch.backends.mps.is_available() else "cpu")
        self.rnn = CTRNN4pop(input_size, hidden_size, **kwargs).to(self.device)  # continuous time RNN
        # self.fc = nn.Linear(hidden_size, output_size, bias=self.bias).to(self.device)  # add linear readout layer
        # init.normal_(self.fc.weight, mean=0, std=0.5)

    def forward(self, x, hidden=None):
        x = x.to(self.device)  # input tensor of shape (Seq Len, Batch, Input Dim)
        rnn_output, _ = self.rnn(x, hidden=hidden)  # get activity of the rnn
        out = torch.stack([
            torch.mean(rnn_output[:, :, :int(self.rnn.hidden_size / 4)], dim=-1),
            torch.mean(rnn_output[:, :, int(self.rnn.hidden_size / 4):int(2*self.rnn.hidden_size / 4)], dim=-1),
            torch.mean(rnn_output[:, :, int(2*self.rnn.hidden_size / 4):int(3*self.rnn.hidden_size / 4)], dim=-1),
            torch.mean(rnn_output[:, :, int(3 * self.rnn.hidden_size / 4):], dim=-1),
            torch.var(rnn_output[:, :, :int(self.rnn.hidden_size / 4)], dim=-1),
            torch.var(rnn_output[:, :, int(self.rnn.hidden_size / 4):int(2*self.rnn.hidden_size / 4)], dim=-1),
            torch.var(rnn_output[:, :, int(2*self.rnn.hidden_size / 4):int(3*self.rnn.hidden_size / 4)], dim=-1),
            torch.var(rnn_output[:, :, int(3 * self.rnn.hidden_size / 4):], dim=-1),
        ], dim=-1)
        # rnn_output_excitatory = rnn_output[:self.number_neurons_excitatory]
        # rnn_output_inhibitory = rnn_output[self.number_neurons_excitatory:]
        # out = torch.stack(
        #     (rnn_output_excitatory.mean(axis=-1), rnn_output_inhibitory.mean(axis=-1),
        #      rnn_output_excitatory.std(axis=-1) ** 2, rnn_output_inhibitory.std(axis=-1) ** 2), dim=-1
        # )
        return out, rnn_output