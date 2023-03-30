'''
This file is used to train the SVM

Before reading through this file, please take a look at dataCollection.py for how we collect training data

We load the training samples from the subdirectory "train_data", files names the format "<lowercase alphabet>.txt"
Each file contains 100 instances (training/testing samples)
We use the first 80 to train the SVM, and the last 20 to test the SVM

The SVM is a classifier, so the letter A is assigned the label 0, B->1, C->2, ..., Z->25
'''

from libsvm.svmutil import *

import numpy as np

import string

def train():
    # each alphabet(26) has 100 recordings of 21 features (feature -> 3D coordinate -> 21*3=63 numbers) each, 80 for train, 20 for test 
    trainData = np.zeros((90*26, 63))
    testData = np.zeros((10*26, 63))

    trainLabels = np.zeros(90*26)
    testLabels = np.zeros(10*26)
    for label in range(26):
        trainLabels[label*90 : (label+1)*90] = label
        testLabels[label*10 : (label+1)*10] = label
    
    # there's files 'a.txt', 'b.txt', .... 'z.txt'
    for i, ch in enumerate(string.ascii_lowercase):
        arr = np.loadtxt('train_data/'+ch+'.txt')
        trainData[i*90 : (i+1)*90] = arr[:90]
        testData[i*10 : (i+1)*10] = arr[90:]

    # print(arr.size)
    # print(arr.shape)
    # for i in range(4):
    #     m = svm_train(trainLabels, trainData, '-t '+str(i)+' -q')

    #     p_label, p_acc, p_val = svm_predict(testLabels, testData, m)

    #     print(p_acc)    
    #     print('*'*100)
    #     print(p_label)    
    #     print('*'*100)

    m = svm_train(trainLabels, trainData, '-t 2 -q')

    p_label, p_acc, p_val = svm_predict(testLabels, testData, m)

    print(p_acc)    
    print('*'*100)
    print(p_label)    
    print('*'*100)

    # print(p_val)    

    # svm_save_model('a2z_model.model', m)

train()