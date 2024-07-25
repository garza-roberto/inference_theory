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
        cell_activity[k] = np.mean(data_raw[k], axis=0)
        cell_activity_all[k] = data_raw[k]

inhibitory_cells = ["pv", "sst", "vip"]
cell_label_list = ["pyr", "pv", "sst", "vip"]

# configuration
n = len(cell_activity.keys())
n_neurons_in_population = 1
size_network = np.ones(n) * n_neurons_in_population  # for now all population have the same size
number_of_probes = len(contrast)
n_connection_to_neuron_i = int(np.sum(size_network))


# %% random generate data
# # initialize randomly arrays of weights (unknown variables, so they will be derived from lstsq)
# W = np.concatenate(([1], np.random.uniform(low=-1, high=1, size=size_network.size)))  # weights in input to neuron i
# K_a = np.random.uniform(low=-1, high=1, size=number_of_probes)  # FF weights in input to neuron i, scaling I
# K_b = np.random.uniform(low=-1, high=1, size=number_of_probes)  # FF weight in input to neuron i, plain offset

# # generate random recording of firing rates and input currents (input current into neuron i, which is specially probed)
# r_i = np.random.uniform(low=0, high=1, size=number_of_probes)  # firing rate of neuron i
# r = np.random.uniform(low=0, high=1, size=(n_connection_to_neuron_i, number_of_probes))  # firing rate of all neurons entering in input to neuron i (including neuron i)
# I = np.random.uniform(low=-1, high=1, size=(1, number_of_probes))  # current in input to neuron i


# %% define and fit system
r = np.zeros((len(cell_activity.keys()), len(contrast)))  # firing rate of all neurons entering in input to neuron i (including neuron i)
for i, k in enumerate(cell_activity.keys()):
    r[i, :] = cell_activity[k]
    if [k for k in cell_activity.keys()][i] in inhibitory_cells:
        r[i, :] *= -1
r = r.T
I = contrast.reshape((1, len(contrast)))
A_0 = np.concatenate((r, I.T, np.ones((number_of_probes, 1))), axis=1)
A_1 = np.concatenate((r, np.zeros((number_of_probes, 1)), np.ones((number_of_probes, 1))), axis=1)

# build matrices
transform_matrix = np.zeros((n, A_0.shape[1]))
ff_matrix = np.zeros((n, 2))
for i, k in enumerate(cell_activity.keys()):
    if k in ["pyr", "pv"]:
        A = A_0
    else:
        A = A_1
    r_i = cell_activity[k]
    y = phi_inverse(r_i)

    # compute nnls
    # x, res, rank, s = np.linalg.lstsq(A.T, y)
    x, rnorm = nnls(A, y)
    print(f"FIT | residual error for cell {k}: {rnorm}")

    transform_matrix[i, :] = x

# %% plot transform matrix
# transform_matrix = np.concatenate((connectivity_matrix, ff_matrix), axis=1)
plt.figure()
plt.pcolormesh(transform_matrix)
plt.colorbar()
# plt.show()


# %% predict and plot results
all_pred = np.zeros((len(cell_activity_all.keys()), number_of_probes))
for i_probe in range(number_of_probes):
    cell_signal = [cell_activity[x][i_probe] * (-1 if x in inhibitory_cells else 1) for x in cell_activity.keys()]
    input_signal = copy.deepcopy(cell_signal)
    input_signal.extend([contrast[i_probe], 1.0])
    r_pred = transform_matrix @ input_signal
    all_pred[:, i_probe] = phi(r_pred)


fig, axs = plt.subplots(1, len(cell_activity_all.keys()))
for i_k, k in enumerate(cell_activity_all.keys()):
    error = np.sqrt(np.sum((all_pred[i_k, :] - cell_activity[k])**2))
    print(f"TEST | residual error for cell {k}: {error}")
    axs[i_k].plot(contrast, cell_activity[k], label="data" if i_k == 0 else "_nolegend_")
    axs[i_k].plot(contrast, all_pred[i_k, :], label="simulation" if i_k == 0 else "_nolegend_")
    axs[i_k].set_title(cell_label_list[i_k])
    axs[i_k].set_xlim(0, 1)
    axs[i_k].set_xlabel("contrast")
    # axs[i_k].set_xticks(contrast)
    axs[i_k].set_ylim(0, 1)
    if i_k == 0:
        axs[i_k].set_ylabel("mean r")
    else:
        axs[i_k].set_yticks([])
fig.legend()
plt.show()