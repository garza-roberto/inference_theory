from scipy.io import loadmat
import numpy as np

path_data = r"C:\Users\Roberto\Academics\courses\cajal_cn_2024\project\inference_theory\data\Data_cell_types_small_size.mat"
data_raw = loadmat(path_data)

pyr = data_raw['pyr']

pyr_mean = np.mean(pyr, axis=0)

