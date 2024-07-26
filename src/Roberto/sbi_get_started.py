# first block in getting started page of the documentation

# %% import dependencies
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

# %% define simulator and prior

num_dim = 3

def simulator(theta):
    # linear gaussian
    return theta + 1.0 + torch.randn_like(theta) * 0.1

prior = BoxUniform(low=-2 * torch.ones(num_dim), high=2 * torch.ones(num_dim))


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


# %% create new observation
theta_true = prior.sample((1,))
# generate our observation
x_obs = simulator(theta_true)

samples = posterior.sample((10000,), x=x_obs)
# _ = pairplot(samples, limits=[[-2, 2], [-2, 2], [-2, 2]], figsize=(6, 6),labels=[r"$\theta_1$", r"$\theta_2$", r"$\theta_3$"])
# plt.show()


# %% test new observation
samples = posterior.sample((10000,), x=x_obs)
_ = pairplot(samples, points=theta_true, limits=[[-2, 2], [-2, 2], [-2, 2]], figsize=(6, 6), labels=[r"$\theta_1$", r"$\theta_2$", r"$\theta_3$"])
plt.show()


# %% check for another random set of parameters
theta_diff = prior.sample((1,))

log_probability_true_theta = posterior.log_prob(theta_true, x=x_obs)
log_probability_diff_theta = posterior.log_prob(theta_diff, x=x_obs)
log_probability_samples = posterior.log_prob(samples, x=x_obs)

print( r'high for true theta :', log_probability_true_theta)
print( r'low for different theta :', log_probability_diff_theta)
print( r'range of posterior samples: min:', torch.min(log_probability_samples),' max :', torch.max(log_probability_samples))
