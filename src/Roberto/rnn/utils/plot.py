import time

from matplotlib import pyplot as plt
from torch import nn, optim


def plot_trial(sample_input, sample_output):
    plt.figure(figsize=(9, 3))
    plt.subplot(1, 2, 1)
    for k in range(sample_input.shape[0]):
        plt.plot(sample_input[k, :], label=sample_input[k, 0], color="red")
    plt.legend()
    plt.xlabel('time')
    plt.subplot(1, 2, 2)
    for j in range(sample_output.shape[0]):
        plt.plot(sample_output[j, :], label=sample_output[k, 0], color="orange")
    plt.legend()
    # plt.show()

def plot_trial_output_target(sample_input, sample_output, sample_target):
    plt.figure(figsize=(9, 3))
    plt.subplot(1, 2, 1)
    for k in range(sample_input.shape[0]):
        plt.plot(sample_input[k, :], label=sample_input[k, 0], color="red")
    plt.legend()
    plt.xlabel('time')
    plt.subplot(1, 2, 2)
    for j in range(sample_output.shape[0]):
        plt.plot(sample_target[j, :], label=f"target" if j==0 else "_nolegend_", color="orange")
        plt.plot(sample_output[j, :], label=f"output" if j==0 else "_nolegend_", color="blue")
    plt.legend()
    # plt.show()