from libsvm.svmutil import *

import numpy as np

import string

def train():
    # each alphabet(26) has 100 recordings of 21 features (feature -> 3D coordinate -> 21*3=63 numbers) each, 80 for train, 20 for test 
    trainData = np.zeros((80*26, 63))
    testData = np.zeros((20*26, 63))

    trainLabels = np.zeros(80*26)
    testLabels = np.zeros(20*26)
    for label in range(26):
        trainLabels[label*80 : (label+1)*80] = label
        testLabels[label*20 : (label+1)*20] = label
    
    # there's files 'a.txt', 'b.txt', .... 'z.txt'
    for i, ch in enumerate(string.ascii_lowercase):
        arr = np.loadtxt('train_data/'+ch+'.txt')
        trainData[i*80 : (i+1)*80] = arr[:80]
        testData[i*20 : (i+1)*20] = arr[80:]

    # print(arr.size)
    # print(arr.shape)
    m = svm_train(trainLabels, trainData)

    p_label, p_acc, p_val = svm_predict(testLabels, testData, m)

    print(p_acc)    

    svm_save_model('a2z_model.model', m)

train()