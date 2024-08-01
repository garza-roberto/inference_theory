# %% dependencies
# check if gpu is available
import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.io import loadmat
from torch.utils.data import DataLoader, random_split

from src.Roberto.rnn.model.rnn_4pop import RNNNet4pop
from src.Roberto.rnn.tasks.moments_task_Npop import moments_task_interpolate
from src.Roberto.rnn.utils.plot import plot_trial, plot_trial_output_target

import matplotlib

from src.Roberto.rnn.utils.train import train_model

matplotlib.use('Qt5Agg')

device = "cpu"# ("mps" if torch.backends.mps.is_available() else "cpu") # *this line is mac M2/M3 specific

# Verify model structure on random input and for one time step


# %% config
batch_size = 20             # size of data batch for training
seq_len = 1000              # sequence length
input_size = 2              # input dimension
output_size = 2             # output dimension
hidden_size = 20           # number of neurons in the recurrent network or "hidden layer"
train_initial_state = True  # whether we want to train the initial state of the network
inhibitory_cells = ["pv", "sst", "vip"]
cell_label_list = ["pyr", "pv", "sst", "vip"]



# %% fetch data and define configurations
path_data = r"C:\Users\Roberto\Academics\courses\cajal_cn_2024\project\inference_theory\data\Data_cell_types_small_size.mat"
data_raw = loadmat(path_data)
contrast = np.squeeze(data_raw['contrast']) / 100
cell_activity = {}
cell_activity_all = {}
cell_activity_array = []
for k in data_raw.keys():
    if not k.startswith("_") and k != "contrast":
        cell_activity[k] = np.mean(data_raw[k], axis=0)
        cell_activity_all[k] = data_raw[k]




# %% define NN
# Create a network instance
model_test = RNNNet4pop(tau=50, input_size=input_size, hidden_size=hidden_size, output_size=output_size, bias=0, train_initial_state=train_initial_state)
model_test = model_test.to(device) # move to gpu
h0 = model_test.rnn.h2h.weight
print('Std of initial connectivity matrix is ' + str(h0.std()))
print('')

# Create some random inputs
I = False # False if no input (zeros) and True if random input to network)
input = I * torch.randn(input_size) * torch.ones(seq_len, batch_size, input_size) # create a constant input of random amplitudes for each unit in the network

# Run the sequence through the rnn network once
out, rnn_output = model_test(input)
print('Input of shape (SeqLen, BatchSize, InputDim)=', input.shape)
print('Output of shape (SeqLen, BatchSize, OutputDim)=', out.shape)
print('Activity of neurons of shape (SeqLen, BatchSize, NumNeurons)=', rnn_output.shape)

print('')
print('Model:')
print(model_test)


# %% preprocess data
task_type = 'moments'

# define parameters for the task *task specific information that is not relevant for the network
cell_activity_sample = {}
config = {
    "r": cell_activity_all,
    "contrasts": contrast,
    "n_neurons": 1e3
}

# create training data set
# dataset = moments_task_subsample(**config)
dataset = moments_task_interpolate(**config)

# Seperate data into training, testing and validation sets
train_size = int(len(dataset) * 0.7) # 70% for training
val_size = int(len(dataset) * 0.15)  # 15% for validation
test_size = len(dataset) - (train_size + val_size)  # Remaining 15% for testing
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size]) # split data

batch_size = 32  # Adjust batch size according to your needs
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=None)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# %% check data
# generate one batch of data when called (for training)
input_sequence, target_sequence, l = next(iter(train_loader)) # Dims: batch x input dim x seq len ;  batch x output dim x seq len; batch x 1

# plot random trial
for n in range(1):
    i = np.random.randint(0, train_size)
    sample_input, sample_output, l = dataset[i]
    print('Plotting sample trial: ' + str(int(i)))
    print('dataset trial length:' + str(l))

    for i_moment in range(2):
        sample_input_np = sample_input.numpy()
        sample_output_np = sample_output.numpy()
        plot_trial(sample_input_np,sample_output_np)


# %% train
# Network Parameters
hidden_size = 40  # number of neurons
input_size = np.array(sample_input).shape[0]  # input dimension
output_size = np.array(sample_output).shape[0]  # output dimension
dt = 1  # 1ms time step
numb_epochs = 100  # number of training epochs (each training epoch run through a batch of data)

# Create an instanciation of the model with the specification of the task above
model = RNNNet4pop(input_size=input_size, hidden_size=hidden_size, output_size=output_size, dt=dt, bias=True, tau=50, train_initial_state=True)
print(model)

continue_training = False  # to avoid re-initialization of the network if training is paused

# run training
model, losses, optimizer = train_model(model, train_loader, numb_epochs=numb_epochs, continue_training=continue_training, weight_decay=1e-5)
model_pred = model

# plot loss during training
plt.figure(figsize=(13, 4))
plt.plot(losses, color="red")
plt.xlabel("Training steps")
plt.title("Losses")

# run training again if loss is still too high
training_stop_threshold = 0.0001  # loss value at which to stop training
if losses[-1] > training_stop_threshold:
    while losses[-1] > training_stop_threshold:
        update_loss = losses[-1]
        print(f"Loss is still high ({losses[-1]:.4f}). Continuing training...")
        additional_epochs = 100  # Define how many more epochs you want to train
        model, losses, optimizer = train_model(model, train_loader, numb_epochs=additional_epochs,
                                               continue_training=True, optimizer=optimizer, losses=losses, weight_decay=1e-5)
        if abs(update_loss - losses[-1]) < 1:  # (max_val*l*0.1/100): # to update if the error is stuck in a certain range
            break
        update_loss = losses[-1]

    # plot loss during training
    plt.figure(figsize=(13, 4))
    plt.plot(losses, color="red")
    plt.xlabel("Training steps")
    plt.title("Losses")

# %% test network after training
# testing config
test_size = 20

# generate testing data
test_dataset = moments_task_interpolate(**config) # create training data set
test_loader = DataLoader(test_dataset, batch_size=test_size, shuffle=False)
inputs, targets, seq_length = next(iter(test_loader)) # create next batch of testing data
inputs = inputs.permute(2, 0, 1).float()
targets = targets.permute(2, 0, 1).float()

# input_rnn_zeros = torch.zeros(seq_len, batch_size, input_size)
outputs, rnn_activity = model(inputs) # get predictions

np.random.seed()
numb_neurons = rnn_activity.shape[2]
r_idx = np.random.choice(np.arange(numb_neurons), 10, replace=False)  # choose 10 random neurons
time_steps = np.arange(inputs.shape[0]) # or 0 depends on above
idx = np.random.randint(0,test_size-1)

print(inputs.shape)
inputs = inputs.cpu().detach().numpy()
targets = targets.cpu().detach().numpy()
outputs = outputs.cpu().detach().numpy()
rnn_activity = rnn_activity.cpu().detach().numpy()

trial_idx = np.random.randint(0,inputs.shape[1]-1) # choose random trial to plot
l = seq_length[0].numpy()
# print(l)
input  = inputs[:, trial_idx, :].T
target = targets[:, trial_idx, :].T
output = outputs[:, trial_idx, :].T

# print(sample_input_np.shape)
# print(input.shape)
config['sample_output'] = output # add network output to config
config['post_training'] = True # add network output to config
plot_trial_output_target(input, output, target)

plt.figure(figsize=(8, 3))
for i in r_idx:
    plt.plot(time_steps, rnn_activity[:,idx,i], label='X Coordinate')
plt.title(f'Activity of neurons for trial {idx}')
plt.xlabel('Time Steps')
plt.ylabel('r(t)')


connectivity_matrix = torch.relu(model.rnn.h2h.weight.data) @ model.rnn.mask
connectivity_matrix_abs = np.abs(connectivity_matrix)

plt.figure(figsize=(8, 8))
plt.matshow(connectivity_matrix_abs)
gridline_positions = np.array([int(i*(model.rnn.hidden_size/4)) for i in range(5)])
label_positions = (gridline_positions[1:] + gridline_positions[:-1]) / 2
plt.hlines(gridline_positions[1:-1]-0.5, xmin=-0.5, xmax=gridline_positions[-1]-0.5, color = 'k', linewidth = 0.5)
plt.vlines(gridline_positions[1:-1]-0.5, ymin=-0.5, ymax=gridline_positions[-1]-0.5, color = 'k',  linewidth = 0.5)
plt.xticks(label_positions, ['E', 'PV', 'SOM', 'VIP'])
plt.yticks(label_positions, ['E', 'PV', 'SOM', 'VIP'])
plt.xlabel('pre-synaptic')
plt.ylabel('post-synaptic')
plt.colorbar()

number_neurons_per_pop = int(model.rnn.hidden_size/4)
pop_matrix = np.zeros((4, 4))
for i_pop in range(4):
    for j_pop in range(4):
        pop_matrix[i_pop, j_pop] = np.nanmean(connectivity_matrix_abs[i_pop*number_neurons_per_pop:(i_pop+1)*number_neurons_per_pop,
                                                           j_pop*number_neurons_per_pop:(j_pop+1)*number_neurons_per_pop])

plt.figure(figsize=(8, 8))
plt.imshow(pop_matrix)
gridline_positions_small = np.array([int(i) for i in range(5)])
label_positions_small = (gridline_positions_small[1:] + gridline_positions_small[:-1]) / 2
plt.hlines(gridline_positions_small[1:-1]-0.5, xmin=-0.5, xmax=gridline_positions_small[-1]-0.5, color = 'k', linewidth = 2)
plt.vlines(gridline_positions_small[1:-1]-0.5, ymin=-0.5, ymax=gridline_positions_small[-1]-0.5, color = 'k',  linewidth = 2)
plt.xticks(label_positions_small-0.5, ['E', 'PV', 'SOM', 'VIP'])
plt.yticks(label_positions_small-0.5, ['E', 'PV', 'SOM', 'VIP'])
plt.xlabel('pre-synaptic')
plt.ylabel('post-synaptic')
cbar = plt.colorbar()
cbar.set_label('connectivity strength', rotation=90)

for layer in model.children():
    try:
        print(layer.state_dict()['weight'])
        print(layer.state_dict()['bias'])
    except KeyError:
        pass



def sigmoid(z):
    return 1/(1 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

def jacobian(rates, weights, tau=1):
    rates = np.array(rates)
    weights = np.array(weights)
    n_pop = weights.shape[0]
    phi_prime_mat = np.diag(sigmoid_prime(rates))
    T_inv = np.diag(np.ones(weights.shape[0])/tau)
    jacobian = T_inv@((phi_prime_mat @ weights) - np.eye(n_pop))
    return jacobian

# plot eigen for matrices WITHOUT CELL TYPES
fig, axs = plt.subplots(1, 1)
J = jacobian(rnn_activity[-1,0,:], connectivity_matrix)
W_eigenvalues, W_eigenvectors = np.linalg.eig(J)
plt.scatter(np.real(W_eigenvalues), np.imag(W_eigenvalues), label=f"all")
W = J
for i_pop in range(4):
    tau = 5 if i_pop == 0 else 1
    selector = np.arange(gridline_positions[i_pop], gridline_positions[i_pop+1])  # .astype(int)
    # print(f"selector: {selector}")
    W_i = np.zeros((len(selector), len(selector)))
    J_i = jacobian(rnn_activity[-1, 0, selector], W_i, tau=tau)
    for i_W, i_s in enumerate(selector):
        for j_W, j_s in enumerate(selector):
            W_i[i_W, j_W] = W[i_s, j_s]
    W_i_eigenvalues, W_i_eigenvectors = np.linalg.eig(W_i)
    axs.scatter(np.real(W_i_eigenvalues), np.imag(W_i_eigenvalues), label=f"only {cell_label_list[i_pop]}")
plt.legend()


# plot eigen for matrices WITHOUT CELL TYPES
fig, axs = plt.subplots(1, 1)
selector = np.arange(number_neurons_per_pop * 4)
for i_pop in range(4):
    tau = 5 if i_pop == 0 else 1
    selector_not = np.arange(gridline_positions[i_pop], gridline_positions[i_pop+1])  # .astype(int)
    selector_i = np.delete(selector, selector_not)
    # print(f"selector: {selector}")
    W_i = np.zeros((number_neurons_per_pop*4 - len(selector_not), number_neurons_per_pop*4 - len(selector_not)))
    J_i = jacobian(rnn_activity[-1, 0, selector_i], W_i, tau=tau)
    for i_W, i_s in enumerate(selector_i):
        for j_W, j_s in enumerate(selector_i):
            W_i[i_W, j_W] = W[i_s, j_s]

    W_i_eigenvalues, W_i_eigenvectors = np.linalg.eig(W_i)
    axs.scatter(np.real(W_i_eigenvalues), np.imag(W_i_eigenvalues), label=f"w/ {cell_label_list[i_pop]}")
    axs.set_xlabel("real")
    axs.set_xlabel("imaginary")
    axs.set_title("Eigenvalues of the Jacobian")
fig.legend()


fig, axs = plt.subplots(4, 4)
for i_pop in range(4):
    for j_pop in range(4):
        selector_i_lim = (gridline_positions[i_pop], gridline_positions[i_pop+1])
        selector_j_lim = (gridline_positions[j_pop], gridline_positions[j_pop+1])
        W_i = connectivity_matrix_abs[selector_i_lim[0]:selector_i_lim[1], selector_j_lim[0]:selector_j_lim[1]]
        W_i_flat = np.array(W_i).flatten()
        hist_values, bin_edges = np.histogram(W_i_flat)
        bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
        axs[i_pop, j_pop].plot(bin_centers, hist_values)  # , label=f"{cell_label_list[i_pop]}-{cell_label_list[j_pop]}")
        axs[i_pop, j_pop].set_xlim([0, 1])
        axs[i_pop, j_pop].set_ylim([0, 80])
        if i_pop == 0:
            axs[i_pop, j_pop].set_xlabel(cell_label_list[j_pop])
        else:
            axs[i_pop, j_pop].set_xticks([])
        if j_pop == 0:
            axs[i_pop, j_pop].set_xlabel(cell_label_list[j_pop])
        else:
            axs[i_pop, j_pop].set_yticks([])
fig.legend()

plt.show()