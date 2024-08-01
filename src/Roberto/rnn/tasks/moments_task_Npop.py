import numpy as np
import torch
from torch.utils.data import Dataset

class moments_task_interpolate(Dataset):
    def __init__(self, r, contrasts, n_neurons, n_samples=1e3, n_steps_input=1e3):
        self.inputs = []
        self.outputs = []
        self.r = np.array(r)
        self.contrasts = np.array(contrasts)
        self.n_neurons = int(n_neurons)
        self.n_samples = int(n_samples)
        self.n_steps_input = int(n_steps_input)

        self.r_mean = np.zeros((len(r.keys()), len(self.contrasts))) # np.mean(r, axis=0)
        self.r_std = np.zeros((len(r.keys()), len(self.contrasts)))  # np.std(r, axis=0)
        self.k_mean = np.zeros((len(r.keys()), 3+1))
        self.k_std = np.zeros((len(r.keys()), 3+1))
        for i_pop, k_pop in enumerate(r.keys()):
            self.r_mean[i_pop] = np.mean(r[k_pop], axis=0)
            self.r_std[i_pop] = np.std(r[k_pop], axis=0)
            self.k_mean[i_pop] = np.polyfit(self.contrasts, self.r_mean[i_pop], 3)
            self.k_std[i_pop] = np.polyfit(self.contrasts, self.r_std[i_pop], 3)

        for i_sample in range(self.n_samples):
            contrast_sample = np.random.uniform(np.min(self.contrasts), np.max(self.contrasts))
            output_constant = np.zeros(len(r.keys())*2)
            for i_pop, k_pop in enumerate(r.keys()):
                mean_sample = np.polyval(self.k_mean[i_pop], contrast_sample)
                std_sample = np.polyval(self.k_std[i_pop], contrast_sample)
                output_constant[i_pop] = mean_sample
                output_constant[len(r.keys()) + i_pop] = std_sample**2
            input = np.ones((1, self.n_steps_input)) * contrast_sample
            output = np.tile(output_constant, (self.n_steps_input, 1))

            ##### pad input and ouput before and after gocue
            input_tensor = torch.from_numpy(input)
            output_tensor = torch.from_numpy(output).permute(1, 0)
            # output_tensor.permute(1, 0)

            self.inputs.append(input_tensor)
            self.outputs.append(output_tensor)
        pass
        # self.outputs = self.outputs.permute(1, 0, 2).float()

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx], self.inputs[idx].shape[0]


class moments_task_interpolate_linda(Dataset):
    def __init__(self, r, contrasts, n_neurons, n_samples=1e4, n_steps_input=1e3):
        self.inputs = []
        self.outputs = []
        self.r = np.array(r)
        self.contrasts = np.array(contrasts)
        self.n_neurons = int(n_neurons)
        self.n_samples = int(n_samples)
        self.n_steps_input = int(n_steps_input)
        res = 5000

        from scipy.interpolate import interp1d
        # Create an interpolating function
        mean_inter = interp1d(contrasts, np.nanmean(r['pyr'],axis=0), kind='cubic')
        var_inter = interp1d(contrasts, np.var(r['pyr'],axis=0), kind='cubic')

        mean_inter2 = interp1d(contrasts, np.nanmean(r['pv'],axis=0), kind='cubic')
        var_inter2 = interp1d(contrasts, np.var(r['pv'],axis=0), kind='cubic')

        for i_sample in range(self.n_samples):
            c = np.random.randint(0,res)/(res)
            input = np.ones((1,self.n_steps_input))*c
            #sampling mean and variance of firing rate

            mean_firing_rate = mean_inter(c)
            mean_variance = var_inter(c)

            mean_firing_rate2 = mean_inter2(c)
            mean_variance2 = var_inter2(c)

            output=np.outer(np.array([mean_firing_rate, mean_variance, mean_firing_rate2, mean_variance2]),np.ones((1,self.n_steps_input)))

            ##### pad input and ouput before and after gocue
            input_tensor = torch.from_numpy(input)
            output_tensor = torch.from_numpy(output)

            self.inputs.append(input_tensor)
            self.outputs.append(output_tensor)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx], self.inputs[idx].shape[0]
