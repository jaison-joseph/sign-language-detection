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

# helper function to generate the per class accuracy using the predicted and expected labels
def perClassAcc(testLabels, predictedLabels):
    lk = {i: [0, 0] for i in set(testLabels)}
    bools = np.equal(testLabels, predictedLabels)
    for i, v in enumerate(bools):
        lk[testLabels[i]][0 if v else 1] += 1
    letters_lk = {chr(ord('A') + int(i)): j for i, j in lk.items()}
    return letters_lk

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

def loadAndTest(model_names, testDataPaths, showPerClassAcc = True):
    
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
        Any iterable sequence that contains a list of the paths of the base directories containing the training samples
        If any of the filePaths are not found, the function will return and not proceed
    trainDataExcludes:
        List of lists of letters to exclude from each file in trainingDataPaths
    testDataPath:
        Path to the base directory containing the testting samples
        If you do not wish to test the model, don't include the argument in the function call or set to None.
        If any of the filePaths are not found, the function will return and not proceed
    modelPath: 
        Path to the directory where the model will be saved, it MUST be created
        By default, set to a folder called models/
        If left empty, will save in the current directory
        If the specified direcotry isn't found, the file will be saved in the current working directory
            In either case, if a file with the provided name already exists, the model will be saved with a different, unique name
    modelName:
        The name to save the newly trained model as.
        If you do not wish to test the model, don't include the argument in the function call or set to None.
        If a file with the same name is found in the provided directory,
            the model will be saved in with another file name that is not present in the current directory
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

'''
one function call to train all models
Please refer to the comments on genericTrain() to know the details
'''
def trainAll():
    genericTrain(
        trainDataPaths = [
            'data/train_data', 'data/train_data_2'
        ],
        testDataPaths = ['data/test_data', 'data/test_data_2'],
        modelPath = 'models/v6',
        modelName = 'a2z_v1_model.model'
    )

    genericTrain(
        trainDataPaths = [
            'data/train_data', 'data/train_data_2', 
            'data/train_data_3'
        ],
        testDataPaths = ['data/test_data', 'data/test_data_2'],
        modelPath = 'models/v6',
        modelName = 'a2z_v2_model.model'
    )

    '''
    the alphabets to remove from train_data_4/ were based on the performance of the v2 model
    '''
    genericTrain(
        trainDataPaths = [
            'data/train_data', 'data/train_data_2', 
            'data/train_data_3', 'data/train_data_4'
        ],
        trainDataExcludes = [
            [],
            [],
            [],
            list(
            set(list('BIJMNSTUVWXZ')) - set(list('JMNSTUXZ'))
            )
        ],
        testDataPaths = ['data/test_data', 'data/test_data_2'],
        modelPath = 'models/v6',
        modelName = 'a2z_v3_model.model'
    )

    '''
    the alphabets to remove from train_data_5/ were based on the performance of the v6 model.
    the same alphabet remove from train
    '''
    genericTrain(
        trainDataPaths = [
            'data/train_data', 'data/train_data_2', 
            'data/train_data_3', 'data/train_data_4', 
            'data/train_data_5'
        ],
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
        testDataPaths = ['data/test_data', 'data/test_data_2'],
        modelPath = 'models/v6',
        modelName = 'a2z_v4_model.model'
    )

    genericTrain(
        trainDataPaths = [
            'data/train_data', 'data/train_data_2', 
            'data/train_data_3', 'data/train_data_4', 
            'data/train_data_5', 'data/train_data_6'
        ],
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
        testDataPaths = ['data/test_data', 'data/test_data_2'],
        modelPath = 'models/v6',
        modelName = 'a2z_v5_model.model'
    )

def testAll():
    loadAndTest(
        [
            'models/v6/a2z_v5_model.model',
            'models/v6/a2z_v6_model.model',
            'models/v6/a2z_v7_model.model',
            'models/v6/a2z_v8_model.model',
            'models/v6/a2z_v9_model.model'
        ],
        ['test_data', 'test_data_2']
    )

trainAll()
# testAll()