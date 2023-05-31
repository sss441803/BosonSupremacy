import numpy as np
from tqdm import tqdm

data = np.fromfile('<replace_with_samples_binary_file>', dtype = 'uint32')
unusedPorts = np.array([1, 2, 3, 4, 5, 6, 7, 8, 17, 36, 60, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192]) - 1
usedPorts = np.array(list(set(list(range(32 * 6))) - set(unusedPorts)))
n_sample = 10 ** 7
data_binary = np.zeros((n_sample, 144), dtype = bool)
for j in tqdm(range(n_sample)):
    out_ = ''
    for i in data[j * 6:6 * (j + 1)]:
        out_ += np.binary_repr(i, width = 32)
    res = np.array(np.array(list(out_), dtype = int)[usedPorts], dtype = bool) # 1 to 144;
    data_binary[j] = res
samples_exp = data_binary[:, ::-1]
np.save('<replace_with_destination_of_experimental_samples>', samples_exp)