import torch
from scipy.io import loadmat
import numpy as np

def phi(x: torch.tensor, a=1, k=2):
    return a*((x) ** k)

path_data = r"C:\Users\Roberto\Academics\courses\cajal_cn_2024\project\inference_theory\data\Data_cell_types_small_size.mat"
data_raw = loadmat(path_data)

A = np.array([[0.85019137, 0.12046141, 0.32576243, 0.,         0.19414135, 0.34042947],
              [0.6267717,  0.,         0.49034239, 0.,         0.38378296, 0.37665983],
              [0.90247262, 0.,         0.,         0.51539458, 0.,         0.3630127 ],
              [0.61244814, 0.13455959, 0.52075599, 0.,         0.,         0.64538931]]).flatten()

# W = A[:, :4]
# W[:, 1:] *= -1

cell_label_list = ["pyr", "pv", "sst", "vip"]
number_populations = 4
number_contrasts = 6
intensity_index = 1
tol = 1e-4
maxiter = 1e4
dt = 0.01
tau = 1

def simulator2(wb):
    # Define inputs
    i = 0
    max_it = 3000

    delta_t = 0.05
    r = torch.rand(24)  # Initialize r with shape (4, 6)
    I_values = data_raw['contrast'][0][:6] / torch.tensor([100])  # Get the 6 different I values

    while i < max_it:

        for j, I in enumerate(I_values):
            A_1 = torch.tensor([r[j * 4], -1 * r[j * 4 + 1], -1 * r[j * 4 + 2], -1 * r[j * 4 + 3], I, 1])
            A_2 = torch.tensor([r[j * 4], -1 * r[j * 4 + 1], -1 * r[j * 4 + 2], -1 * r[j * 4 + 3], 0, 1])

            r_delta = -r[j * 4:j * 4 + 4] + phi(
                torch.tensor([A_1 @ wb[0:6], A_1 @ wb[6:12], A_2 @ wb[12:18], A_2 @ wb[18:24]]))
            r[j * 4:j * 4 + 4] = r[j * 4:j * 4 + 4] + delta_t * r_delta

        eps = torch.sum(torch.pow(r_delta, 2))
        if eps < 1e-4 or torch.isnan(eps):
            break

        i += 1

    return r

r = simulator2(A)

print(r)

