'''
This file is used to train the SVM

Before reading through this file, please take a look at dataCollection.py for how we collect training data

We load the training samples from the subdirectory "train_data", files names the format "<lowercase alphabet>.txt"
Each file contains 100 instances (training/testing samples)
We use the first 80 to train the SVM, and the last 20 to test the SVM

The SVM is a classifier, so the letter A is assigned the label 0, B->1, C->2, ..., Z->25
'''

from libsvm.svmutil import *
from libsvm.svm import *

import numpy as np

import string

import os

import glob

# to drop into the python repl
import code

def perClassAcc(testLabels, predictedLabels):
    lk = {i: [0, 0] for i in set(testLabels)}
    bools = np.equal(testLabels, predictedLabels)
    for i, v in enumerate(bools):
        lk[testLabels[i]][0 if v else 1] += 1
    letters_lk = {chr(ord('A') + int(i)): j for i, j in lk.items()}
    return letters_lk


# takes in first 300 features from the frist file of each alphabets samples
def train_v2():
    trainData = np.zeros((300*26, 63))
    testData = np.zeros((10*26, 63))

    trainLabels = np.zeros(300*26)
    testLabels = np.zeros(10*26)
    for label in range(26):
        trainLabels[label*300 : (label+1)*300] = label
        testLabels[label*10 : (label+1)*10] = label
    
    # there's files 'a.txt', 'b.txt', .... 'z.txt'
    for i, ch in enumerate(string.ascii_uppercase):
        filePath = os.path.join(
            os.getcwd(),
            'train_data',
            ch
        )
        fileName = os.listdir(filePath)[0]
        filePath = os.path.join(filePath, fileName)
        print('loading '+ch)
        trainData[i*300 : (i+1)*300] = np.loadtxt(filePath)[:300]

    for i, ch in enumerate(string.ascii_lowercase):
        arr = np.loadtxt('train_data/'+ch+'.txt')
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

    m = svm_train(trainLabels, trainData, '-t 0 -q')

    p_label, p_acc, p_val = svm_predict(testLabels, testData, m)

    print(p_acc)    
    print('*'*100)
    print(p_label)    
    print('*'*100)

    print('done')

    # print(p_val)    

    # svm_save_model('a2z_v2_model.model', m)

# takes in any # of samples from the first file of each alphabet's samples
def train_v3():
    trainData = np.zeros((1, 63))
    # stores the number of training samples for each alphabet: key: alphabet (a->0, b->1), value: # of samples
    lengths = {}

    testData = np.zeros((10*26, 63))
    testLabels = np.zeros(10*26)
    
    # there's files 'a.txt', 'b.txt', .... 'z.txt'
    for i, ch in enumerate(string.ascii_uppercase):
        filePath = os.path.join(
            os.getcwd(),
            'train_data',
            ch
        )
        fileName = os.listdir(filePath)[0]
        filePath = os.path.join(filePath, fileName)
        print('loading '+ch)
        arr = np.loadtxt(filePath)
        lengths[i] = arr.shape[0]
        trainData = np.concatenate(
            (trainData, arr),
            axis = 0
        )

    trainData = trainData[1:]

    trainLabels = np.concatenate([np.ones(v)*k for k, v in lengths.items()], axis=0)

    # load up old test data & test labels
    for i, ch in enumerate(string.ascii_lowercase):
        arr = np.loadtxt('train_data/'+ch+'.txt')
        testData[i*10 : (i+1)*10] = arr[-10:]
        testLabels[i*10 : (i+1)*10] = i

    m = svm_train(trainLabels, trainData, '-t 0 -q')

    p_label, p_acc, p_val = svm_predict(testLabels, testData, m)

    print(p_acc)    
    print('*'*100)
    print(p_label)    
    print('*'*100)

    # svm_save_model('a2z_v3_model.model', m)

def train_v4():

    # a/A -> 0, b/B -> 1, ... z/Z: 25
    label_lk_ = {ch: n for n, ch in enumerate(string.ascii_lowercase)} | {ch: n for n, ch in enumerate(string.ascii_uppercase)} 

    # load the training data & create the training data labels 
    train_file_paths = glob.glob('train_data_2/' + '*/*.txt')

    labels = [label_lk_[i[i.index('\\')+1]] for i in train_file_paths]

    trainData = [
        np.loadtxt(f) for f in train_file_paths
    ]    

    trainLabels = np.concatenate([
        np.ones(trainData[i].shape[0])*label for i, label in enumerate(labels)
    ])

    trainData = np.concatenate(trainData, axis = 0)


    # load the test data & create the test data labels
    test_file_paths = glob.glob('test_data/' + '*/*.txt')

    labels = [label_lk_[i[i.index('\\')+1]] for i in test_file_paths]

    testData = [
        np.loadtxt(f) for f in test_file_paths
    ]    

    testLabels = [
        np.ones(testData[i].shape[0])*label for i, label in enumerate(labels)
    ]

    # for foo in testLabels:
    #     print(foo.shape)
    
    testLabels = np.hstack(testLabels)

    testData = np.concatenate(testData, axis = 0)

    print(trainData.shape)
    print(trainLabels.shape)
    print(testData.shape)
    print(testLabels.shape)


    # train model
    m = svm_train(trainLabels, trainData, '-t 0 -q')

    p_label, p_acc, p_val = svm_predict(testLabels, testData, m)

    print(p_acc)    
    print('*'*100)
    # print(p_label)    
    # print('*'*100)


    # save model
    svm_save_model('a2z_v4_model.model', m)

'''

a function to load the data from a sub directory name provided as an argument

assumes that the file structure of the training & testing files are as follows:

    test_data/
        A/
            <bla bla bla>.txt
            <bla bla bla>.txt
            <bla bla bla>.txt
        B/
            <bla bla bla>.txt
            <bla bla bla>.txt

        ....

        Z/
            <bla bla bla>.txt

returns two ndarrays, data & labels (data.shape[0] == labels.shape[0], labels is a 1D array)
'''
def getDataAndLabels(folderName, removeLetters = []):
    label_lk_ = {ch: n for n, ch in enumerate(string.ascii_lowercase)} | {ch: n for n, ch in enumerate(string.ascii_uppercase)} 

    # load the training data & create the training data labels 
    file_paths = glob.glob(folderName + '/*/*.txt')

    if len(file_paths) == 0:
        print(f'In call to getDataAndLabels(folderName = {folderName}, removeLetters = {removeLetters}), no files found')
        return None, None

    labels = [label_lk_[i[i.index('\\')+1]] for i in file_paths]

    try:
        data = [
            np.loadtxt(f) for f in file_paths
        ]   
    except ValueError:
        print(f'In call to getDataAndLabels(folderName = {folderName}, removeLetters = {removeLetters}), file format of one of the .txt files found was not a ndarray ')
        return None, None 

    labels = np.concatenate([
        np.ones(data[i].shape[0])*label for i, label in enumerate(labels)
    ])

    data = np.concatenate(data, axis = 0)

    if len(removeLetters) != 0:
        removeLetterNums = list(map(
            lambda x: ord(x) - ord('A') if x.isupper() else ord(x) - ord('a'), 
            removeLetters
        ))
        keepIdxs = [i for i, j in enumerate(labels) if j not in removeLetterNums]
        data = data[keepIdxs]
        labels = labels[keepIdxs]

    return data, labels

def train_v5():
    trainData_1, trainLabels_1 = getDataAndLabels('train_data')
    trainData_2, trainLabels_2 = getDataAndLabels('train_data_2')

    trainData = np.concatenate(
        (trainData_1, trainData_2)
    )
    trainLabels = np.concatenate(
        (trainLabels_1, trainLabels_2)
    )

    trainData_1, trainLabels_1, trainData_2, trainLabels_2 = None, None, None, None

    testData, testLabels = getDataAndLabels('test_data')

    # train model
    m = svm_train(trainLabels, trainData, '-t 0 -q')

    p_label, p_acc, p_val = svm_predict(testLabels, testData, m)

    print(p_acc)    
    print('*'*100)

    # save model
    svm_save_model('a2z_v5_model.model', m)

def train_v6():
    trainData_1, trainLabels_1 = getDataAndLabels('train_data')
    trainData_2, trainLabels_2 = getDataAndLabels('train_data_2')
    trainData_3, trainLabels_3 = getDataAndLabels('train_data_3')

    trainData = np.concatenate(
        (trainData_1, trainData_2, trainData_3)
    )
    trainLabels = np.concatenate(
        (trainLabels_1, trainLabels_2, trainLabels_3)
    )

    trainData_1, trainLabels_1, trainData_2, trainLabels_2, trainData_3, trainLabels_3 = None, None, None, None, None, None

    testData, testLabels = getDataAndLabels('test_data')

    # train model
    m = svm_train(trainLabels, trainData, '-t 0 -q')

    p_label, p_acc, p_val = svm_predict(testLabels, testData, m)

    print(p_acc)    
    print('*'*100)

    # save model
    svm_save_model('a2z_v6_model.model', m)

def train_v7():
    trainData_1, trainLabels_1 = getDataAndLabels('train_data')
    trainData_2, trainLabels_2 = getDataAndLabels('train_data_2')
    trainData_3, trainLabels_3 = getDataAndLabels('train_data_3')
    trainData_4, trainLabels_4 = getDataAndLabels('train_data_4')

    trainData = np.concatenate(
        (trainData_1, trainData_2, trainData_3, trainData_4)
    )
    trainLabels = np.concatenate(
        (trainLabels_1, trainLabels_2, trainLabels_3, trainLabels_4)
    )

    trainData_1, trainLabels_1, trainData_2, trainLabels_2 = None, None, None, None
    trainData_3, trainLabels_3, trainData_4, trainLabels_4 = None, None, None, None

    testData, testLabels = getDataAndLabels('test_data')

    # train model
    m = svm_train(trainLabels, trainData, '-t 0 -q')

    p_label, p_acc, p_val = svm_predict(testLabels, testData, m)

    print(p_acc)    
    print('*'*100)

    # save model
    svm_save_model('a2z_v7_model.model', m)

def train_v8():
    trainData_1, trainLabels_1 = getDataAndLabels('train_data')
    trainData_2, trainLabels_2 = getDataAndLabels('train_data_2')
    trainData_3, trainLabels_3 = getDataAndLabels('train_data_3')
    trainData_4, trainLabels_4 = getDataAndLabels('train_data_4')
    trainData_5, trainLabels_5 = getDataAndLabels('train_data_5')

    trainData = np.concatenate(
        (trainData_1, trainData_2, trainData_3, trainData_4, trainData_5)
    )
    trainLabels = np.concatenate(
        (trainLabels_1, trainLabels_2, trainLabels_3, trainLabels_4, trainLabels_5)
    )

    trainData_1, trainLabels_1, trainData_2, trainLabels_2 = None, None, None, None
    trainData_3, trainLabels_3, trainData_4, trainLabels_4 = None, None, None, None
    trainData_5, trainLabels_5 = None, None

    testData, testLabels = getDataAndLabels('test_data')

    # train model
    m = svm_train(trainLabels, trainData, '-t 0 -q')

    p_label, p_acc, p_val = svm_predict(testLabels, testData, m)

    print(p_acc)    
    print('*'*100)

    class_acc = perClassAcc(testLabels, p_label)
    inaccurates = {i: j for i, j in class_acc.items() if j[1] != 0}
    list(map(print, inaccurates.items()))
    print('*'*100)

    # save model
    folderPath = './' + 'models'
    if not os.path.exists(folderPath):
        os.mkdir(folderPath)
    svm_save_model('models/' + 'a2z_v8_model.model', m)

def train_v9():
    trainData_1, trainLabels_1 = getDataAndLabels('train_data')
    trainData_2, trainLabels_2 = getDataAndLabels('train_data_2')
    trainData_3, trainLabels_3 = getDataAndLabels('train_data_3')
    trainData_4, trainLabels_4 = getDataAndLabels('train_data_4')
    trainData_5, trainLabels_5 = getDataAndLabels('train_data_5')
    trainData_6, trainLabels_6 = getDataAndLabels('train_data_6')

    trainData = np.concatenate(
        (trainData_1, trainData_2, trainData_3, trainData_4, trainData_5, trainData_6)
    )
    trainLabels = np.concatenate(
        (trainLabels_1, trainLabels_2, trainLabels_3, trainLabels_4, trainLabels_5, trainLabels_6)
    )

    trainData_1, trainLabels_1, trainData_2, trainLabels_2 = None, None, None, None
    trainData_3, trainLabels_3, trainData_4, trainLabels_4 = None, None, None, None
    trainData_5, trainLabels_5, trainData_6, trainLabels_6 = None, None, None, None

    testData, testLabels = getDataAndLabels('test_data')

    # train model
    m = svm_train(trainLabels, trainData, '-t 0 -q')

    p_label, p_acc, p_val = svm_predict(testLabels, testData, m)

    print(p_acc)    
    print('*'*100)

    # save model
    svm_save_model('a2z_v9_model.model', m)

def train_v10():
    trainData_1, trainLabels_1 = getDataAndLabels('train_data', ['J', 'N', 'X', 'Y'])
    trainData_2, trainLabels_2 = getDataAndLabels('train_data_2')
    trainData_3, trainLabels_3 = getDataAndLabels('train_data_3')
    trainData_4, trainLabels_4 = getDataAndLabels('train_data_4')
    trainData_5, trainLabels_5 = getDataAndLabels('train_data_5')
    trainData_6, trainLabels_6 = getDataAndLabels('train_data_6')

    trainData = np.concatenate(
        (trainData_1, trainData_2, trainData_3, trainData_4, trainData_5, trainData_6)
    )
    trainLabels = np.concatenate(
        (trainLabels_1, trainLabels_2, trainLabels_3, trainLabels_4, trainLabels_5, trainLabels_6)
    )

    trainData_1, trainLabels_1, trainData_2, trainLabels_2 = None, None, None, None
    trainData_3, trainLabels_3, trainData_4, trainLabels_4 = None, None, None, None
    trainData_5, trainLabels_5, trainData_6, trainLabels_6 = None, None, None, None

    testData, testLabels = getDataAndLabels('test_data')

    # train model
    m = svm_train(trainLabels, trainData, '-t 0 -q')

    p_label, p_acc, p_val = svm_predict(testLabels, testData, m)
    print(p_acc)    
    print('*'*100)
    class_acc = perClassAcc(testLabels, p_label)
    inaccurates = {i: j for i, j in class_acc.items() if j[1] != 0}
    list(map(print, inaccurates.items()))
    print('*'*100)

    # save model
    svm_save_model('a2z_v10_model.model', m)

def loadAndTest(model_name, testFolderName, showPerClassAcc = True):
    m = svm_load_model(model_name)
    testData, testLabels = getDataAndLabels(testFolderName)
    p_label, p_acc, p_val = svm_predict(testLabels, testData, m)
    print(p_acc)    
    print('*'*100)
    if showPerClassAcc:
        class_acc = perClassAcc(testLabels, p_label)
        inaccurates = {i: j for i, j in class_acc.items() if j[1] != 0}
        list(map(print, inaccurates.items()))
        print('*'*100)

def loadAndTest_v2(model_names, testDataPaths, showPerClassAcc = True):
    
    allData = [getDataAndLabels(p) for p in testDataPaths]
    testData = np.concatenate([i[0] for i in allData])
    testLabels = np.concatenate([i[1] for i in allData])
    allData = None
    for mName in model_names:
        m = svm_load_model(mName)
        p_label, p_acc, p_val = svm_predict(testLabels, testData, m)

        print(p_acc)    
        print('*'*100)

        if showPerClassAcc:
            class_acc = perClassAcc(testLabels, p_label)
            inaccurates = {i: j for i, j in class_acc.items() if j[1] != 0}
            list(map(print, inaccurates.items()))
            print('*'*100)

def testFoo():
    print('whole set')
    data, labels = getDataAndLabels('test_data')
    print(len(data))
    print(set(labels))
    print('*' * 100)
    print('after removing the b\'s ')
    data, labels = getDataAndLabels('test_data', ['b'])
    print(len(data))
    print(set(labels))

# a function to crash, on purpose
# you get to know what exceptions to catch!
def crashFunction():
    a,b = getDataAndLabels('lol')
    print(a)
    print(b)
    # filePath = os.path.join(
    #     os.getcwd(),
    #     'lol'
    # )
    # fileNames = os.listdir(filePath)
    # print(fileNames)

# crashFunction()

'''
single function to call to train, test and save model

input:
    trainDataPaths:
        any iterable sequence that contains a list of the paths of the base directories containing the training samples
        if any of the filePaths are not found, the function will return and not proceed
    trainDataExcludes:
        list of lists of letters to exclude from each file in trainingDataPaths
    testDataPath:
        path to the base directory containing the testting samples
        if any of the filePaths are not found, the function will return and not proceed
    modelPath: 
        path to the directory where the model will be saved, it MUST be created
        By default, set to a folder called models/
        If left empty, will save in the current directory
        If the specified direcotry isn't found, the file will be saved in the current working directory
            In either case, if a file with the provided name already exists, the model will be saved with a different, unique name
    modelName:
        the name to save the newly trained model as
        If a file with the same name is found in the provided directory,
        the model will be saved in with another file name that is not present in
        the current directory
'''
def genericTrain(
    trainDataPaths,
    trainDataExcludes = [],
    testDataPaths = None,
    modelPath = 'models',
    modelName = None
):
    
    shouldTest = (testDataPaths is not None)
    shouldSave = (modelName is not None)

    # check if empty, returns true if not empty
    if (trainDataExcludes):
        # check if mismatch in lengths of trainingDataPaths and trainingDataExcludes
        if len(trainDataExcludes) != len(trainDataPaths):
            print(f'lengths of trainDataPaths ({len(trainDataPaths)}) does not match length of trainDataExcludes ({len(trainDataExcludes)})')
            return
        allData = [getDataAndLabels(p, trainDataExcludes[i]) for i, p in enumerate(trainDataPaths)]
    else:
        allData = [getDataAndLabels(p) for p in trainDataPaths]
    trainData = np.concatenate([i[0] for i in allData])
    trainLabels = np.concatenate([i[1] for i in allData])
    allData = None

    if trainData is None:
        print('Could not load data properly')
        return
    if trainLabels is None:
        print('Could not create labels properly')
        return
    
    if shouldTest:
        allData = [getDataAndLabels(p) for p in testDataPaths]
        testData = np.concatenate([i[0] for i in allData])
        testLabels = np.concatenate([i[1] for i in allData])
        allData = None
        if testData is None:
            print('Could not laod data properly')
            return
        if testLabels is None:
            print('Could not create labels properly')
            return
    
    m = svm_train(trainLabels, trainData, '-t 0 -q')

    if shouldTest:
        p_label, p_acc, p_val = svm_predict(testLabels, testData, m)

        print(p_acc)    
        print('*'*100)

        class_acc = perClassAcc(testLabels, p_label)
        inaccurates = {i: j for i, j in class_acc.items() if j[1] != 0}
        list(map(print, inaccurates.items()))
        print('*'*100)

    if shouldSave:
        
        # remove trailing '/'
        if modelPath[-1] == '/':
            modelPath = modelPath[:-1]
        
        # get all files in the directory we're gonna save in
        filePath = os.path.join(
            os.getcwd(),
            modelPath
        )
        # if the 
        try:
            fileNames = os.listdir(filePath)
        except FileNotFoundError:
            print('the specified path to save the model could not be found/accessed. Will save to current directory')
            modelPath = ''
            fileNames = os.listdir(os.getcwd())


        # strip out the '.model' extension if it was added
        if '.model' in modelName:
            noExt = modelName[:modelName.index('.model')]
        else:
            noExt = modelName
        
        # keep trying a file that isn't present in the directory
        if noExt + '.model' in fileNames:
            splits = [noExt, '_', '0']
            noExt = ''.join(splits)
            while noExt + '.model' in fileNames:
                splits[-1] = str(int(splits[-1])+1)
                noExt = ''.join(splits)
            print(f'Warning: {splits[0]}.model will be saved as {noExt}.model because the original file name already exists.')

        finalFileName = noExt + '.model'
        svm_save_model(modelPath + '/' + finalFileName, m)

def testAll():
    loadAndTest('models/v2/a2z_model.model', 'test_data')
    loadAndTest('models/v2/a2z_v2_model.model', 'test_data')
    loadAndTest('models/v2/a2z_v3_model.model', 'test_data')
    loadAndTest('models/v2/a2z_v4_model.model', 'test_data')
    loadAndTest('models/v2/a2z_v5_model.model', 'test_data')
    loadAndTest('models/v2/a2z_v6_model.model', 'test_data')
    loadAndTest('models/v3/a2z_v7_model.model', 'test_data')
    loadAndTest('models/v3/a2z_v8_model.model', 'test_data')
    loadAndTest('models/v3/a2z_v9_model.model', 'test_data')

def testAll_v2():
    loadAndTest_v2(
        [
            'models/v2/a2z_v1_model.model',
            'models/v2/a2z_v2_model.model',
            'models/v2/a2z_v3_model.model',
            'models/v2/a2z_v4_model.model',
            'models/v2/a2z_v5_model.model',
            'models/v2/a2z_v6_model.model',
            'models/v3/a2z_v7_model.model',
            'models/v3/a2z_v8_model.model',
            'models/v3/a2z_v9_model.model'
        ],
        ['test_data', 'test_data_2']
    )

def trainAll():
    # the training dataset is tiny, so the testing dataset is smaller as well
    genericTrain(
        trainDataPaths = ['train_data_old'],
        testDataPath = ['test_data'],
        modelPath = 'models/v5',
        modelName = 'a2z_v1_model.model'
    )

    # the training dataset is tiny, so the testing dataset is smaller as well
    genericTrain(
        trainDataPaths = ['train_data'],
        testDataPath = ['test_data'],
        modelPath = 'models/v5',
        modelName = 'a2z_v2_model.model'
    )

    # the training dataset is tiny, so the testing dataset is smaller as well
    genericTrain(
        trainDataPaths = ['train_data_old', 'train_data'],
        testDataPath = ['test_data'],
        modelPath = 'models/v5',
        modelName = 'a2z_v3_model.model'
    )

    # the training dataset is tiny, so the testing dataset is smaller as well
    genericTrain(
        trainDataPaths = ['train_data_2'],
        testDataPath = ['test_data'],
        modelPath = 'models/v5',
        modelName = 'a2z_v4_model.model'
    )

    genericTrain(
        trainDataPaths = ['train_data', 'train_data_2'],
        testDataPaths = ['test_data', 'test_data_2'],
        modelPath = 'models/v5',
        modelName = 'a2z_v5_model.model'
    )

    genericTrain(
        trainDataPaths = ['train_data', 'train_data_2', 'train_data_3'],
        testDataPaths = ['test_data', 'test_data_2'],
        modelPath = 'models/v5',
        modelName = 'a2z_v6_model.model'
    )

    genericTrain(
        trainDataPaths = ['train_data', 'train_data_2', 'train_data_3', 'train_data_4'],
        trainDataExcludes = [
            [],
            [],
            [],
            list(
            set(list('BIJMNSTUVWXZ')) - set(list('JMNSTUXZ'))
            )
        ],
        testDataPaths = ['test_data', 'test_data_2'],
        modelPath = 'models/v5',
        modelName = 'a2z_v7_model.model'
    )

    genericTrain(
        trainDataPaths = ['train_data', 'train_data_2', 'train_data_3', 'train_data_4', 'train_data_5'],
        trainDataExcludes = [
            [],
            [],
            [],
            list(
            set(list('BIJMNSTUVWXZ')) - set(list('JMNSTUXZ'))
            ),
            list(
                set(list('IJMNRSTUVXYZ')) - set(list('IJMNSTUXZ'))
            )
        ],
        testDataPaths = ['test_data', 'test_data_2'],
        modelPath = 'models/v5',
        modelName = 'a2z_v8_model.model'
    )

    genericTrain(
        trainDataPaths = ['train_data', 'train_data_2', 'train_data_3', 'train_data_4', 'train_data_5', 'train_data_6'],
        trainDataExcludes = [
            [],
            [],
            [],
            list(
                set(list('BIJMNSTUVWXZ')) - set(list('JMNSTUXZ'))
            ),
            list(
                set(list('IJMNRSTUVXYZ')) - set(list('IJMNSTUXZ'))
            ),
            list(
                set(list('JMNSTUXYZ')) - set(list('JMNSXZ'))
            )
        ],
        testDataPaths = ['test_data', 'test_data_2'],
        modelPath = 'models/v5',
        modelName = 'a2z_v9_model.model'
    )

trainAll()
# testAll_v2()