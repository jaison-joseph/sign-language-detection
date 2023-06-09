import os
from string import ascii_uppercase
import numpy as np
import glob

# bigSum = 0

folders = [
    'train_data',
    'train_data_2',
    'train_data_3',
    'train_data_4',
    'train_data_5',
    'train_data_6'
]

for f in folders:
    checker = set()
    for ch in ascii_uppercase:
        # filePath = os.path.join(
        #     os.getcwd(),
        #     '..',
        #     'test_data',
        #     'A'
        # )
        file_paths = glob.glob(f'{f}/{ch}/*.txt')
        data = [np.loadtxt(f) for f in file_paths]
        foo = sum([d.shape[0] for d in data])
        checker.add(foo)
        # if remaining != 0:
        # print(ch, remaining)
        # bigSum += remaining
    print(f, checker)

# list(map(print, file_paths))

# print(a)
# print(bigSum)