import numpy as np
import torch
from torch.utils.data import Dataset


class fdr_task(Dataset):
    def __init__(self,N=1000,nT=1000, **kwargs):
        self.inputs = []
        self.outputs = []
        self.N  = kwargs.get('N', 1000)
        self.nT = kwargs.get('nT', 1000)

        for _ in range(N):

            this_n=np.abs(np.random.randn())
            input=np.ones((1,nT))*this_n
            output=np.outer(np.array([this_n**2,this_n**3]),np.ones((1,nT)))

            ##### pad input and ouput before and after gocue
            input_tensor = torch.from_numpy(input)
            output_tensor = torch.from_numpy(output)

            self.inputs.append(input_tensor)
            self.outputs.append(output_tensor)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx], self.inputs[idx].shape[1]