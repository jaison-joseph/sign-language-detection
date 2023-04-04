from collections import deque
import bisect
import string
import numpy as np
import os
from datetime import datetime

'''
This is a class written to manage the feature collection and handling, as well as storage

Allows for deletion of store elements from any index 
'''
class LL():
    '''
    Constructor
    Params:
        - sz: the numer of recods to store
        - recordDims: the dimensions of a single record
    '''
    def __init__(self, sz, recordDims) -> None:
        # the maximum size of the store
        self.size_ = sz

        # the store variable itself
        self.store_ = np.zeros([sz] + recordDims)

        # a dictionary that tells for each index, if it's occupied or not
        # 1-> vacant | 0 -> empty
        self.index_status_ = {i: 1 for i in range(sz)}

        # keeps count of the # of vacant spaces in the store
        self.vacantCount_ = sz
        
        # the labels for each of the records in the store
        self.labels_ = ['' for _ in range(sz)]

    '''
    adds a record x, with label y, to the store
    will not do anything if store is full
    '''
    def add(self, x, y):

        # check to see if the store is empty
        if self.vacantCount_ == 0:
            print('Store full')
            return
        
        # we know that the store must have at least one empty index, so we use next to find it
        idx = next((k for k, v in self.index_status_.items() if v == 1))

        # storing the record & label
        self.store_[idx] = x
        self.labels_[idx] = y

        # set the index idx status to occupied
        self.index_status_[idx] = 0

        # updating vacant count
        self.vacantCount_ -= 1

    def delete(self, idx):
        # checking if the index is valid
        if idx not in self.index_status_:
            print("Invalid index provided")
            return
        
        # check if the index is empty, return if empty
        if self.index_status_[idx] == 1:
            print("Index provided is empty")
            return
        
        # set the provided index to vacant
        self.index_status_[idx] = 1

        # clear the label associated with the index
        self.labels_[idx] = ''

        # update the counter for vacant slots in the store
        self.vacantCount_ += 1

    def saveContents(self):
        # check to see if the store is empty
        if self.vacantCount_ == self.size_:
            print('no contents to save')
        
        for alphabet in set(self.labels_):
            # get all keys (keys are indices of the store) that are occupied & whose associated label equals 'alphabet
            idxs = [k for k, v in self.index_status_.items() if v == 0 and self.labels_[k] == alphabet]
            
            folderPath = './train_data/'+alphabet
            if not os.path.exists(folderPath):
                os.mkdir(folderPath)
            fileName = alphabet + datetime.now().strftime("%m_%d_%y %H-%M-%S") + ".txt"
            np.savetxt(
                folderPath + '/' + fileName, 
                np.concatenate([self.store_[i] for i in idxs])
            )

    def printInternals(self):
        print('index_status_: ', self.index_status_)
        # print('Store: ', self.store_)
        print('Labels: ', self.labels_)
        print('\n\n ------------------------------------------------------------ \n\n')

recordDims = [2, 3]
l = LL(10, recordDims)
for i, ch in enumerate(string.ascii_uppercase[:10]):
    l.add(np.random.rand(2, 3), ch)

l.printInternals()

l.delete(-10)   # shouldn't work
l.delete(6)
l.printInternals()

l.add(np.random.rand(2, 3), 'B')
l.printInternals()

l.delete(0)
l.delete(1)
l.add(np.random.rand(2, 3), 'E')
l.printInternals()

l.add(np.random.rand(2, 3), 'E')
l.printInternals()

l.saveContents()
