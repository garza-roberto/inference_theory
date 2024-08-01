# %% dependencies
import copy

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import nnls
from scipy.io import loadmat

import matplotlib
matplotlib.use('Qt5Agg')


# %% utils
# transfer function
def phi(x, a=1, k=2):
    return a*(x**k)

# inverse transfer function
def phi_inverse(y, a=1, k=2):
    return (y / a)**(1/k)

# %% fetch data and define configurations
path_data = r"C:\Users\Roberto\Academics\courses\cajal_cn_2024\project\inference_theory\data\Data_cell_types_small_size.mat"
data_raw = loadmat(path_data)
contrast = np.squeeze(data_raw['contrast']) / 100
cell_activity = {}
cell_activity_all = {}
for k in data_raw.keys():
    if not k.startswith("_") and k != "contrast":
        cell_activity[k] = {}
        cell_activity[k]["mean"] = np.mean(data_raw[k], axis=0)
        cell_activity[k]["std"] = np.std(data_raw[k], axis=0)

        cell_activity_all[k] = data_raw[k]

inhibitory_cells = ["pv", "sst", "vip"]
cell_label_list = ["pyr", "pv", "sst", "vip"]
number_found_items = 0

# configuration
n_simulations = int(5e4)
n = len(cell_activity.keys())
n_neurons_in_population = 1
size_network = np.ones(n) * n_neurons_in_population  # for now all population have the same size
number_of_probes = len(contrast)
n_connection_to_neuron_i = int(np.sum(size_network))
tau = 1
rnorm_threshold = 0.18  # 0.5  #


# %% random generate data
# # initialize randomly arrays of weights (unknown variables, so they will be derived from lstsq)
# W = np.concatenate(([1], np.random.uniform(low=-1, high=1, size=size_network.size)))  # weights in input to neuron i
# K_a = np.random.uniform(low=-1, high=1, size=number_of_probes)  # FF weights in input to neuron i, scaling I
# K_b = np.random.uniform(low=-1, high=1, size=number_of_probes)  # FF weight in input to neuron i, plain offset

# # generate random recording of firing rates and input currents (input current into neuron i, which is specially probed)
# r_i = np.random.uniform(low=0, high=1, size=number_of_probes)  # firing rate of neuron i
# r = np.random.uniform(low=0, high=1, size=(n_connection_to_neuron_i, number_of_probes))  # firing rate of all neurons entering in input to neuron i (including neuron i)
# I = np.random.uniform(low=-1, high=1, size=(1, number_of_probes))  # current in input to neuron i

rnorm_list = []
A_list = []
for i_simulation in range(n_simulations):
    r = np.zeros((len(cell_activity.keys()),
                  len(contrast)))  # firing rate of all neurons entering in input to neuron i (including neuron i)
    for i, k in enumerate(cell_activity.keys()):
        r[i, :] = np.abs(np.random.normal(loc=cell_activity[k]["mean"], scale=cell_activity[k]["std"]))
        if [k for k in cell_activity.keys()][i] in inhibitory_cells:
            r[i, :] *= -1

    # %% define and fit system
    r = r.T
    I = contrast.reshape((1, len(contrast)))
    A_0 = np.concatenate((r, I.T, np.ones((number_of_probes, 1))), axis=1)
    A_1 = np.concatenate((r, np.zeros((number_of_probes, 1)), np.ones((number_of_probes, 1))), axis=1)

    # build matrices
    transform_matrix = np.zeros((n, A_0.shape[1]))
    ff_matrix = np.zeros((n, 2))
    rnorm = np.zeros(len(cell_activity.keys()))
    for i, k in enumerate(cell_activity.keys()):
        if k in ["pyr", "pv"]:
            A = A_0
        else:
            A = A_1
        r_i = np.abs(r[:, i])
        y = phi_inverse(r_i)

        # compute nnls
        # x, res, rank, s = np.linalg.lstsq(A.T, y)
        x, rnorm_i = nnls(A, y)
        # print(f"FIT | residual error for cell {k}: {rnorm}")
        rnorm[i] = rnorm_i
        transform_matrix[i, :] = x
    rnorm_list.append(np.linalg.norm(rnorm))

    if ((1 - transform_matrix[0, :] @ phi(r[:, 0])) / tau < 0) and \
        transform_matrix[2, 2] == 0 and transform_matrix[3, 3] == 0 and \
        transform_matrix[0, 3] <= 0.5 and transform_matrix[1, 3] <= 0.5  and \
        (np.linalg.norm(rnorm) < rnorm_threshold):
        number_found_items += 1
        print(f"FOUND {number_found_items}")
        # print(transform_matrix)
        A_list.append(transform_matrix)

print(np.mean(rnorm_list))
print(np.std(rnorm_list))
print(np.quantile(rnorm_list, q=0.01))


A_array = np.array(A_list)
A_mean = np.mean(A_array, axis=0)
print(f"A_mean: {A_mean}")

fig = plt.figure()
plt.matshow(np.mean(A_array, axis=0))
plt.colorbar()
plt.show()
