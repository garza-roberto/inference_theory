# %% import dependencies
import numpy as np
import torch
from matplotlib import pyplot as plt
from sbi.analysis import pairplot
from sbi.inference import SNPE, simulate_for_sbi
from sbi.utils import BoxUniform
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)
from scipy.io import loadmat
from scipy import stats

import matplotlib
matplotlib.use('Qt5Agg')

# %% configuration
number_of_populations = 4
number_of_parameters = 6
number_of_probes = 6
number_of_simulations = 1e4
inhibitory_cells = ["pv", "sst", "vip"]



# %% utils
# transfer function
def phi(x, a=1, k=2):
    return a*(x**k)


# %% fetch data and process them
path_data = r"C:\Users\Roberto\Academics\courses\cajal_cn_2024\project\inference_theory\data\Data_cell_types_small_size.mat"
data_raw = loadmat(path_data)
contrast = np.squeeze(data_raw['contrast']) / 100
cell_activity = {}
cell_activity_all = {}
for k in data_raw.keys():
    if not k.startswith("_") and k != "contrast":
        cell_activity[k] = np.mean(data_raw[k], axis=0)
        cell_activity_all[k] = data_raw[k]
r = np.zeros((len(cell_activity.keys()), len(contrast)))  # firing rate of all neurons entering in input to neuron i (including neuron i)
for i, k in enumerate(cell_activity.keys()):
    r[i, :] = cell_activity[k]
    if [k for k in cell_activity.keys()][i] in inhibitory_cells:
        r[i, :] *= -1
r = r.T
I = contrast.reshape((1, len(contrast)))
A_0 = np.concatenate((r, I.T, np.ones((number_of_probes, 1))), axis=1)
A_1 = np.concatenate((r, np.zeros((number_of_probes, 1)), np.ones((number_of_probes, 1))), axis=1)


# A = np.concatenate((r, I.T, np.ones((number_of_probes, 1))), axis=1)
# A_1 = np.concatenate((r, np.zeros((number_of_probes, 1)), np.ones((number_of_probes, 1))), axis=1)


# %% prior
prior = BoxUniform(low=0 * torch.ones(number_of_parameters), high=1 * torch.ones(number_of_parameters))
# prior = [torch.distributions.exponential.Exponential(torch.tensor([1.0])) for _ in range(number_of_parameters)]

cell_list = [c for c in cell_activity.keys()]
# cell_list = [cell_list[0]]
fig, axs = plt.subplots(1, len(cell_list))
for i, k in enumerate(cell_list):
    if k in ["pyr", "pv"]:
        A = A_0
    else:
        A = A_1

    A = torch.from_numpy(np.array(A, dtype=np.float32))

    # %% simulator
    def simulator(theta):
        return phi(torch.matmul(A, theta))

    # %% process elements
    # Check prior, return PyTorch prior.
    prior, num_parameters, prior_returns_numpy = process_prior(prior)

    # Check simulator, returns PyTorch simulator able to simulate batches.
    simulator = process_simulator(simulator, prior, prior_returns_numpy)

    # Consistency check after making ready for sbi.
    check_sbi_inputs(simulator, prior)


    # %% initialize inference object
    inference = SNPE(prior=prior)


    # %% simulate
    theta, x = simulate_for_sbi(simulator, proposal=prior, num_simulations=2000)
    print("theta.shape", theta.shape)
    print("x.shape", x.shape)


    # %% add simulation to our inference
    inference = inference.append_simulations(theta, x)


    # %% train
    density_estimator = inference.train()


    # %% get posterior
    posterior = inference.build_posterior(density_estimator)
    print(posterior)  # prints how the posterior was trained


    # %% condition on firing rates
    # generate our observation
    x_obs = cell_activity[k]

    samples = posterior.sample((10000,), x=x_obs)
    _ = pairplot(samples, figsize=(6, 6))


    # best_param = torch.tensor(np.array(stats.mode(np.array(samples), axis=0))[0, :])
    # prediction = simulator(best_param)
    test_param = posterior.sample((100,), x=x_obs)
    prediction = simulator(test_param)

    try:
        axs[i].plot(contrast, cell_activity[k], label="data")
        for i_p in range(prediction.shape[0]):
            axs[i].plot(contrast, prediction[i_p, :], label="simulation", color="k", alpha=0.1, linewidth=1)
            axs[i].set_title(k)
            axs[i].set_xlim(0, 1)
            axs[i].set_xlabel("contrast")
            # axs[i_k].set_xticks(contrast)
            axs[i].set_ylim(0, 1)
            if i == 0:
                axs[i].set_ylabel("mean r")
            else:
                axs[i].set_yticks([])
    except:
        axs.plot(contrast, cell_activity[k], label="data")
        for i_p in range(prediction.shape[0]):
            axs.plot(contrast, prediction[i_p, :], label="simulation", color="k", alpha=0.1, linewidth=1)
        axs.set_title(k)
        axs.set_xlim(0, 1)
        axs.set_xlabel("contrast")
        # axs[i_k].set_xticks(contrast)
        axs.set_ylim(0, 1)
        if i == 0:
            axs.set_ylabel("mean r")
        else:
            axs.set_yticks([])

plt.show()



