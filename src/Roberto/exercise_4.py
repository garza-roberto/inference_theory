# %% dependencies
# check if gpu is available
import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.io import loadmat
from torch.utils.data import DataLoader, random_split

from src.Roberto.rnn.moments_task import moments_task_subsample, moments_task_interpolate
from src.Roberto.rnn.rnn_1pop import RNNNet
from src.Roberto.rnn.utils import plot_trial, train_model

device = "cpu"# ("mps" if torch.backends.mps.is_available() else "cpu") # *this line is mac M2/M3 specific

# Verify model structure on random input and for one time step


# %% config
batch_size = 20             # size of data batch for training
seq_len = 1000              # sequence length
input_size = 2              # input dimension
output_size = 2             # output dimension
hidden_size = 500           # number of neurons in the recurrent network or "hidden layer"
train_initial_state = True  # whether we want to train the initial state of the network
inhibitory_cells = ["pv", "sst", "vip"]
cell_label_list = ["pyr", "pv", "sst", "vip"]



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




# %% define NN
# Create a network instance
model_test = RNNNet(tau=50, input_size=input_size, hidden_size=hidden_size, output_size=output_size, bias=False, train_initial_state=train_initial_state)
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
config = {
    "r": cell_activity_all[cell_label_list[0]],
    "contrasts": contrast,
    "n_neurons": 1e4
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
hidden_size = 50  # number of neurons
input_size = np.array(sample_input).shape[0]  # input dimension
output_size = np.array(sample_output).shape[0]  # output dimension
dt = 1  # 1ms time step
numb_epochs = 1000  # number of training epochs (each training epoch run through a batch of data)

# Create an instanciation of the model with the specification of the task above
model = RNNNet(input_size=input_size, hidden_size=hidden_size, output_size=output_size, dt=dt, bias=False, tau=50,
               train_initial_state=True)
print(model)

continue_training = False  # to avoid re-initialization of the network if training is paused

# run training
model, losses, optimizer = train_model(model, train_loader, numb_epochs=numb_epochs,
                                       continue_training=continue_training)
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
        additional_epochs = 2000  # Define how many more epochs you want to train
        model, losses, optimizer = train_model(model, train_loader, numb_epochs=additional_epochs,
                                               continue_training=True, optimizer=optimizer, losses=losses)
        if abs(update_loss - losses[
            -1]) < 1:  # (max_val*l*0.1/100): # to update if the error is stuck in a certain range
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
test_dataset = moments_task_subsample(**config) # create training data set
test_loader = DataLoader(test_dataset, batch_size=test_size, shuffle=False)
inputs, targets, seq_length = next(iter(test_loader)) # create next batch of testing data
inputs = inputs.permute(2, 0, 1).float()
targets = targets.permute(2, 0, 1).float()

# input_rnn_zeros = torch.zeros(seq_len, batch_size, input_size)
outputs, rnn_activity = model(inputs) # get predictions

np.random.seed()
numb_neurons = rnn_activity.shape[2]
r_idx = np.random.choice(np.arange(numb_neurons), 10, replace=False) # choose 10 random neurons
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
plot_trial(input,target)
plot_trial(input,output)

plt.figure(figsize=(8, 3))
for i in r_idx:
    plt.plot(time_steps, rnn_activity[:,idx,i], label='X Coordinate')
plt.title(f'Activity of neurons for trial {idx}')
plt.xlabel('Time Steps')
plt.ylabel('r(t)')
plt.show()