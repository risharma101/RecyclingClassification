import numpy as np
import os
import random
import feature

TRAININGFEATUREPATH = "../featureextractor/featureInfo/training/"
VALIDATIONFEATUREPATH = "../featureextractor/featureInfo/validation/"

# Get the training batch
def getTrainingBatch(batchSize):
    trainingPath = os.listdir(TRAININGFEATUREPATH)
    data = []
    labels = []
    random.shuffle(trainingPath)
    if batchSize > len(trainingPath):
        batchSize = len(trainingPath)

    for featureVector in trainingPath[:batchSize]:
        loadFeatureVecotr = np.load("{0}{1}".format(TRAININGFEATUREPATH, featureVector))
        featureData = loadFeatureVector[:, 1:]
        data = np.concatenate((data, featureData), axis=0) if np.array(data).size else featureData
        currLabel = feature.getCatFromName(featureVector)
        labelArr = np.full(len(featureData), currLabel, dtype=int)
        labels = np.concatenate((labels, labelArr), axis=0)

    return np.array(data), np.array(labels)


# Get the validation batch
def getTestingBatch(batchSize=5):
    testingPath = os.listdir(VALIDATIONFEATUREPATH)
    data = []
    labels = []
    test = 0
    random.shuffle(testingPath)
    if batchSize > len(testingPath):
        batchSize = len(testingPath)

    for featureVector in testingPath[:batchSize]:
        # The first column of all these is the label so we can extract that
        loadFeatureVector = np.load("{0}{1}".format(VALIDATIONFEATUREPATH, featureVector))
        featureLabel = [x[0] for x in loadFeatureVector]
        featureData = loadFeatureVector[:, 1:]
        data = np.concatenate((data, featureData), axis=0) if np.array(data).size else featureData
        currLabel = feature.getCatFromName(featureVector)
        #labelArr = np.empty(len(featureData))
        labelArr = np.full(len(featureData),currLabel, dtype=int)
        labels = np.concatenate((labels, labelArr), axis=0)
        test = featureVector
    print(np.array(data).shape)
    print(np.array(labels).shape)
    return np.array(data), np.array(labels)

getTestingBatch()
