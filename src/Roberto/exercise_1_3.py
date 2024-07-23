import numpy as np
from scipy.optimize import nnls


# transfer function
def phi(x, a=1, k=2):
    return a*(x**k)

# inverse transfer function
def phi_inverse(y, a=1, k=2):
    return (y / a)**(1/k)

# configuration
n = 2
n_neurons_in_population = 100
size_network = np.ones(n) * n_neurons_in_population  # for now all population have the same size
number_of_probes = 100
n_connection_to_neuron_i = int(np.sum(size_network))

# initialize rnadomly arrays of weights (unknown variables, so they will be derived from lstsq)
W = np.concatenate(([1], np.random.uniform(low=-1, high=1, size=size_network.size)))  # weights in input to neuron i
K_a = np.random.uniform(low=-1, high=1, size=number_of_probes)  # FF weights in input to neuron i, scaling I
K_b = np.random.uniform(low=-1, high=1, size=number_of_probes)  # FF weight in input to neuron i, plain offset

# generate random recording of firing rates and input currents (input current into neuron i, which is specially probed)
r_i = np.random.uniform(low=0, high=1, size=number_of_probes)  # firing rate of neuron i
r = np.random.uniform(low=0, high=1, size=(n_connection_to_neuron_i, number_of_probes))  # firing rate of all neurons entering in input to neuron i (including neuron i)
I = np.random.uniform(low=-1, high=1, size=(1, number_of_probes))  # current in input to neuron i

# build matrices
A = np.concatenate((r, I, np.ones((1, number_of_probes))), axis=0)
y = phi_inverse(r_i)

# compute lstsq
# x, res, rank, s = np.linalg.lstsq(A.T, y)
x, rnorm = nnls(A.T, y)
W = x[:n_connection_to_neuron_i]  # weights in input to neuron i
K = x[n_connection_to_neuron_i:]  # FF weights in input to neuron i
