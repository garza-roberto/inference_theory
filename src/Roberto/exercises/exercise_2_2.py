# %% import dependencies
import copy

import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.io import loadmat

import matplotlib
matplotlib.use('Qt5Agg')


# configurations
cell_label_list = ["pyr", "pv", "sst", "vip"]
number_populations = 4
number_contrasts = 6
intensity_index = 1
tol = 1e-4
maxiter = 1e4
dt = 0.01
tau = 1

# %% prior
# initialize from nnls (my)
# A = np.array([[0.85019137, 0.12046141, 0.32576243, 0.,         0.19414135, 0.34042947],
#               [0.6267717,  0.,         0.49034239, 0.,         0.38378296, 0.37665983],
#               [0.90247262, 0.,         0.,         0.51539458, 0.,         0.3630127 ],
#               [0.61244814, 0.13455959, 0.52075599, 0.,         0.,         0.64538931]])

A = np.array([[0.75506452, 0.,         1.27550021, 0.44641837, 0.56890152, 0.64719746],
 [0.28576243, 0.,         0.,         0.02022892, 0.1645415,  0.46370091],
 [0.41679125, 0.00167599, 0.22971568, 0.46178917, 0.,         0.53643092],
 [0.53922099, 0.,         1.46167525, 0.,         0.,         0.78358848]])
A = A  # / number_populations  # np.sqrt(number_populations)
W = A[:, :4]
W[:, 1:] *= -1
A[0:4, 0:4] = W

def phi_derivative(x):
    return 2*x

W_eigenvalues, W_eigenvectors = np.linalg.eig(W)

# get mean recordings
path_data = r"/data/Data_cell_types_small_size.mat"
data_raw = loadmat(path_data)
contrast_list = np.squeeze(data_raw['contrast']) / 100
# cell_activity = np.zeros((number_populations, number_contrasts))
# i = 0
# for k in data_raw.keys():
#     if not k.startswith("_") and k != "contrast":
#         cell_activity[i, :] = np.mean(data_raw[k], axis=0)
#         i += 1

# get steady state of the network
def phi(x, a=1, k=2):
    return a*((np.heaviside(x, 0) * x) ** k)

def simulator_wrapper(theta):
    rates = np.zeros((number_populations, number_contrasts))

    for i_c, c in enumerate(contrast_list):
        rates[:, i_c] = copy.deepcopy(simulator(theta, c))

    # rates = torch.flatten(rates)

    return rates

def simulator(theta, c):
    # theta = torch.from_numpy(theta)
    # theta[2:, 4] = 0
    # theta[:, 1:4] *= -1

    r = np.zeros(number_populations)
    # r = np.random.random(number_populations)
    I = c
    # print(I)

    error = np.inf
    iteration = 0
    while error > tol:
        iteration += 1
        if iteration > maxiter:
            print("maxiter threshold hit")
            break
        # print(f" START | iter: {iteration} | r: {r_sign}")
        x = np.concatenate((r, np.array([I, 1.0])))
        dr = (-r + phi(theta @ x)) * dt / tau
        r = r + dr
        error = np.linalg.norm(dr)
        # if np.isnan:
        # print(f"BOOM! Simulation exploded at iteration {iteration}")
    print(f"iterations: {iteration}")
    return r

def jacobian(rates, weights, taus = np.ones(4)):
    n_pop = weights.shape[0]
    phi_prime_mat = np.diag(phi_derivative(rates))
    T_inv = np.diag(1/taus)
    jacobian = T_inv@((phi_prime_mat @ weights) - np.eye(n_pop))
    return jacobian

r = simulator_wrapper(A)
# print(r)

# phi_prime = phi_derivative(cell_activity[:, intensity_index])
# phi_prime = phi_derivative(r)

# J = -1 * np.eye(W.shape[0]) + np.diag(phi_prime) @ W
J = jacobian(r, W)
print(f"r: {r}")
print(f"A: {A}")
print(f"W: {W}")
print(f"J: {J}")
J_eigenvalues, J_eigenvectors = np.linalg.eig(J)

plt.figure()
fig, axs = plt.subplots(1, number_contrasts)
plt.scatter(np.real(J_eigenvalues), np.imag(J_eigenvalues), label="original")

for i in range(number_populations):
    tau = 5 if i == 0 else 1
    for j in range(number_contrasts):
        plot_section = axs[j]
        selector = [x for x in range(W.shape[0]) if x != i]
        # print(f"selector: {selector}")
        W_i = np.zeros((len(selector), len(selector)))
        for i_W, i_s in enumerate(selector):
            for j_W, j_s in enumerate(selector):
                W_i[i_W, j_W] = W[i_s, j_s]
        # print(f"W_i: {W_i}")
        # phi_prime_i = phi_prime[selector, j]
        # print(f"phi_prime_i: {phi_prime_i}")
        # J_i = -1 * np.eye(W_i.shape[0]) + np.diag(phi_prime_i) @ W_i / tau

        J_i = jacobian(r[selector, j], W_i, taus=np.ones(len(selector)))

        J_i_eigenvalues, J_i_eigenvectors = np.linalg.eig(J_i)
        plot_section.scatter(np.real(J_i_eigenvalues), np.imag(J_i_eigenvalues), label=f"w/o {cell_label_list[i]}, I={contrast_list[j]}")

        # print(f"J_i: {J_i}")
        # print(f"J_i_eigenvalues: {J_i_eigenvalues}")

        plot_section.legend()
plt.show()


