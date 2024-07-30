# %% import dependencies
import copy

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
number_of_simulations = 1e6
tol = 1e-4
maxiter = 1e4
dt = 1e-2
tau = 1
inhibitory_cells = ["pv", "sst", "vip"]



# %% utils
# transfer function
# def phi(x, a=1, k=2):
#     return a*(x**k)

def phi(x: torch.tensor, a=1, k=2):
    return a*((torch.heaviside(x, torch.tensor([0.0])) * x) ** k)


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
# initialize from nnls
# prior_mean = np.array([[0.85019137, 0.12046141, 0.32576243, 0.,         0.19414135, 0.34042947],
#               [0.6267717,  0.,         0.49034239, 0.,         0.38378296, 0.37665983],
#               [0.90247262, 0.,         0.,         0.51539458, 0.,         0.3630127 ],
#               [0.61244814, 0.13455959, 0.52075599, 0.,         0.,         0.64538931]]).flatten()
# prior = [torch.distributions.normal.Normal(loc=prior_mean[i], scale=0.5) for i in range(len(prior_mean))]
# prior = [BoxUniform(low=torch.zeros(1) * np.max([0, prior_mean[i]-0.5])
#                     , high=(prior_mean[i] + 0.5) * torch.ones(1)) for i in range(len(prior_mean))]
prior = [torch.distributions.exponential.Exponential(torch.tensor([4.0]))] * (number_of_populations * number_of_parameters)

cell_list = [c for c in cell_activity.keys()]
cell_list = [cell_list[0]]
contrast_list = contrast
fig, axs = plt.subplots(1, number_of_populations)

# %% simulator
def simulator_wrapper(theta):
    rates = torch.zeros((number_of_populations, number_of_probes))

    for i_c, c in enumerate(contrast_list):
        rates[:, i_c] = simulator(theta, c)

    if len(rates.shape) > 2:
        rates = rates.reshape((rates.shape[0], rates.shape[1] * rates.shape[2]))
    else:
        rates = torch.flatten(rates)

    return rates

def simulator(theta, c):
    if len(theta.shape) > 1:
        theta = theta.reshape((theta.shape[0], number_of_populations, number_of_parameters))
        theta[:, 2:, 4] = 0
        theta[:, :, 1:4] *= -1
    else:
        theta = theta.reshape((number_of_populations, number_of_parameters))
        theta[2:, 4] = 0
        theta[:, 1:4] *= -1

    r = torch.zeros(number_of_populations)
    I = c

    error = np.inf
    iteration = 0
    while error > tol:
        iteration += 1
        if iteration > maxiter:
            print("maxiter threshold hit")
            break
        x = torch.from_numpy(np.concatenate((r, np.array([I, 1.0])), dtype=np.float32))
        dr = (-r + phi(torch.matmul(theta, x))) * dt / tau
        r = r + dr
        error = torch.norm(dr)
    return r

# %% process elements
# Check prior, return PyTorch prior.
prior, num_parameters, prior_returns_numpy = process_prior(prior)

# Check simulator, returns PyTorch simulator able to simulate batches.
simulator_wrapper = process_simulator(simulator_wrapper, prior, prior_returns_numpy)

# Consistency check after making ready for sbi.
check_sbi_inputs(simulator_wrapper, prior)


# %% initialize inference object
inference = SNPE(prior=prior)


# %% simulate
theta, x = simulate_for_sbi(simulator_wrapper, proposal=prior, num_simulations=int(number_of_simulations))  # ,
                            # simulation_batch_size=100, num_workers=4)
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
x_obs = np.array([cell_activity[cell] for cell in cell_activity.keys()])

samples = posterior.sample((10000,), x=x_obs)
_ = pairplot(samples, figsize=(6, 6))


# best_param = torch.tensor(np.array(stats.mode(np.array(samples), axis=0))[0, :])
# prediction = simulator(best_param)
test_param = posterior.sample((30,), x=x_obs)
prediction = simulator(test_param)

print(f"x_obs: {x_obs}")
print(f"prediction: {prediction}")

for i_c, c in enumerate(cell_activity.keys()):
    plot_section = axs[i_c]
    plot_section.plot(contrast, cell_activity[c], label="data" if i_c == 0 else "_nolegend_")
    for i_p in range(prediction.shape[0]):
        plot_section.plot(contrast, prediction[i_p, i_c, :], label="simulation" if i_c == 0 and i_p == 0 else "_nolegend_", color="k", alpha=0.1, linewidth=1)
        plot_section.set_title(c)
        plot_section.set_xlim(0, 1)
        plot_section.set_xlabel("contrast")
        # axs[i_k].set_xticks(contrast)
        plot_section.set_ylim(0, 1)
        if i_c == 0:
            plot_section.set_ylabel("rates")
        else:
            plot_section.set_yticks([])

fig.legend()
fig.show()



