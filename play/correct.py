import numpy as np

import glob

import os

file_paths = glob.glob(f'train_data_3/*/*.txt')
# data = [np.loadtxt(f) for f in file_paths]
print(len(file_paths))
file_names = list(map(os.path.basename, file_paths))
for name, path in zip(file_names, file_paths):
    data = np.loadtxt(path)
    if data.shape[0] > 2000:
        data = data[:2000]
        np.savetxt(
            path,
            data
        )
# np.savetxt(
#     fileName, 
#     np.concatenate([self.store_[i] for i in idxs])
# )