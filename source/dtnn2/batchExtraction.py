import numpy as np
import os
import random
import feature

# TRAININGFEATUREPATH = "../featureextractor/featureInfo/training/"
# VALIDATIONFEATUREPATH = "../featureextractor/featureInfo/validation/"

# # Get the training batch
# def getTrainingBatch(batchSize):
#     trainingPath = os.listdir(TRAININGFEATUREPATH)
#     data = []
#     labels = []
#     random.shuffle(trainingPath)
#     if batchSize > len(trainingPath):
#         batchSize = len(trainingPath)

#     for featureVector in trainingPath[:batchSize]:
#         loadFeatureVector = np.load("{0}{1}".format(TRAININGFEATUREPATH, featureVector))
#         featureData = loadFeatureVector[:, 1:]
#         data = np.concatenate((data, featureData), axis=0) if np.array(data).size else featureData
#         currLabel = feature.getCatFromName(featureVector)
#         labelArr = np.full(len(featureData), currLabel, dtype=int)
#         labels = np.concatenate((labels, labelArr), axis=0)

#     return np.array(data), np.array(labels)

PCADATAPATH = '../featureextractor/featureInfoPCA/pca_data.npy'
LDADATAPATH = '../featureextractor/featureInfoPCA/lda_data.npy'

def splitData():
    data = np.load(LDADATAPATH)
    featureData = data[:, 1:]
    featureLabels = data[:, 0]
    temp = list(zip(featureData, featureLabels))
    random.shuffle(temp)
    dataArr, labelArr = zip(*temp)
    trainingData = dataArr[:int(len(dataArr)*.8)]
    trainingLabels = labelArr[:int(len(labelArr)*.8)]
    validationData = dataArr[:int(len(dataArr)*.8)]
    validationLabels = labelArr[:int(len(labelArr)*.8)]

    return np.array(trainingData), np.array(trainingLabels), np.array(validationData), np.array(validationLabels)

def getTrainingBatch(trainingData, trainingLabels, batchSize):
    temp = list(zip(trainingData, trainingLabels))
    random.shuffle(temp)
    data, label = zip(*temp)
    data = np.array(data)
    label = np.array(label)
    return data[:batchSize], label[:batchSize]

# Get the validation batch
# def getTestingBatch(batchSize=20):
#     testingPath = os.listdir(VALIDATIONFEATUREPATH)
#     data = []
#     labels = []
#     test = 0
#     random.shuffle(testingPath)
#     if batchSize > len(testingPath):
#         batchSize = len(testingPath)

#     for featureVector in testingPath[:batchSize]:
#         #print(featureVector)
#         # The first column of all these is the label so we can extract that
#         loadFeatureVector = np.load("{0}{1}".format(VALIDATIONFEATUREPATH, featureVector))
#         featureLabel = [x[0] for x in loadFeatureVector]
#         featureData = loadFeatureVector[:, 1:]
#         data = np.concatenate((data, featureData), axis=0) if np.array(data).size else featureData
#         currLabel = feature.getCatFromName(featureVector)
#         #labelArr = np.empty(len(featureData))
#         labelArr = np.full(len(featureData),currLabel, dtype=int)
#         labels = np.concatenate((labels, labelArr), axis=0)
#         test = featureVector
#     #print(np.array(data).shape)
#     #print(np.array(labels).shape)
#     return np.array(data), np.array(labels)

#getTestingBatch()
