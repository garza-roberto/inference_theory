import numpy as np
import torch
from torch.utils.data import Dataset


class moments_task_subsample(Dataset):
    def __init__(self, r, contrasts, n_neurons, n_samples=1e4, n_in_sample=1e3, n_steps_input=1e3):  #, N=1000, nT=1000, **kwargs):
        self.inputs = []
        self.outputs = []
        # self.N  = kwargs.get('N', 1000)
        # self.nT = kwargs.get('nT', 1000)
        self.r = np.array(r)
        self.contrasts = np.array(contrasts)
        self.n_neurons = int(n_neurons)
        self.n_samples = int(n_samples)
        self.n_steps_input = int(n_steps_input)


        for i_sample in range(self.n_samples):
            i_contrast_sample = np.random.randint(len(self.contrasts))
            contrast_sample = self.contrasts[i_contrast_sample]
            index_list_sample = np.random.randint(0, self.r[:, i_contrast_sample].shape[0], int(n_in_sample))
            dataset_sample = self.r[index_list_sample, i_contrast_sample]
            input = np.ones((1, self.n_steps_input)) * contrast_sample
            output = np.array([np.mean(dataset_sample)*np.ones(self.n_steps_input), np.ones(self.n_steps_input)*(np.std(dataset_sample)**2)])

            ##### pad input and ouput before and after gocue
            input_tensor = torch.from_numpy(input)
            output_tensor = torch.from_numpy(output)

            self.inputs.append(input_tensor)
            self.outputs.append(output_tensor)

    def __len__(self):
        return self.r.shape[0]

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx], self.inputs[idx].shape[0]

class moments_task_interpolate(Dataset):
    def __init__(self, r, contrasts, n_neurons, n_samples=1e4, n_steps_input=1e3):
        self.inputs = []
        self.outputs = []
        self.r = np.array(r)
        self.contrasts = np.array(contrasts)
        self.n_neurons = int(n_neurons)
        self.n_samples = int(n_samples)
        self.n_steps_input = int(n_steps_input)

        self.r_mean = np.mean(r, axis=0)
        self.r_std = np.std(r, axis=0)
        k_mean = np.polyfit(self.contrasts, self.r_mean, 3)
        k_std = np.polyfit(self.contrasts, self.r_std, 3)
        for i_sample in range(self.n_samples):
            contrast_sample = np.random.uniform(np.min(self.contrasts), np.max(self.contrasts))
            mean_sample = np.polyval(k_mean, contrast_sample)
            std_sample = np.polyval(k_std, contrast_sample)
            input = np.ones((1, self.n_steps_input)) * contrast_sample
            output = np.array([mean_sample*np.ones(self.n_steps_input), np.ones(self.n_steps_input)*(std_sample ** 2)])

            ##### pad input and ouput before and after gocue
            input_tensor = torch.from_numpy(input)
            output_tensor = torch.from_numpy(output)

            self.inputs.append(input_tensor)
            self.outputs.append(output_tensor)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx], self.inputs[idx].shape[0]
