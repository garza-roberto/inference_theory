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


def train_model(model, train_loader, numb_epochs, continue_training=False, optimizer=None, losses=[], bias=False):
    # if training was paused
    if not continue_training or optimizer is None:
        optimizer = optim.Adam(model.parameters(),
                               lr=0.1)  # define optimization (pytorch tools). "lr" is learning rate for adam optimizer
        losses = []  # to save losses during training

    criterion = nn.MSELoss()  # criterion = nn.CrossEntropyLoss()
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [200,500,1000,2000,5000,7000,10000,15000,20000], gamma = 0.5) # learning rate scheduler

    running_loss = 0  # initialize running loss
    start_time = time.time()
    print('Training network...')
    for i in range(numb_epochs):  # loop over epochs/over training batches; calls different batches of the data

        # create next batch of training data
        inputs, target, l = next(iter(train_loader))

        # reshape data
        inputs = inputs.permute(2, 0, 1).float()  # Input has shape (SeqLen, Batch, Dim)
        target = target.permute(2, 0, 1).float()  # Target has shape (SeqLen, Batch,Dim)
        # target = target.permute(1, 0, 2).float()  # Target has shape (SeqLen, Batch,Dim)
        inputs = inputs.to('cpu')  # inputs.to('mps')
        target = target.to('cpu')  # target.to('mps')

        ## reshape target accordingly!

        optimizer.zero_grad()  # zero the gradient buffers
        output, _ = model(inputs)  # get output of the network
        output = output.float()  # Reshape to (SeqLen x Batch, OutputSize)
        loss = criterion(output, target)  # make sure everything is of the same size

        losses.append(loss.item())
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 20) # if you want to clip
        optimizer.step()
        # scheduler.step() # to uncomment if you want to use optimization scheduler

        # Compute the running loss every 100 steps
        running_loss += loss.item()
        if i % 100 == 99:
            running_loss /= 100
            print('Step {}, Loss {:0.4f}, Time {:0.1f}s'.format(
                i + 1, running_loss, time.time() - start_time))
            running_loss = 0
    return model, losses, optimizer