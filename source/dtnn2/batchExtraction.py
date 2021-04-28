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

SAVEPATH = 'splitData/pcaSplitData'

def splitData():
    data = np.load(PCADATAPATH)
    featureData = data[:, 1:]
    featureLabels = data[:, 0]
    temp = list(zip(featureData, featureLabels))
    random.shuffle(temp)
    dataArr, labelArr = zip(*temp)
    trainingData = dataArr[:int(len(dataArr)*.8)]
    trainingLabels = labelArr[:int(len(labelArr)*.8)]
    validationData = dataArr[:int(len(dataArr)*.8)]
    validationLabels = labelArr[:int(len(labelArr)*.8)]

    np.save('{0}/training_data.npy'.format(SAVEPATH), np.array(trainingData))
    np.save('{0}/training_labels.npy'.format(SAVEPATH), np.array(trainingLabels))
    np.save('{0}/validation_data.npy'.format(SAVEPATH), np.array(validationData))
    np.save('{0}/validation_labels.npy'.format(SAVEPATH), np.array(validationLables))

def loadData():
    trainingData = np.load('./{0}/training_data.npy'.format(SAVEPATH))
    trainingLabels = np.load('./{0}/training_labels.npy'.format(SAVEPATH))
    validationData = np.load('./{0}/validation_data.npy'.format(SAVEPATH))
    validationLabels = np.load('./{0}/validation_labels.npy'.format(SAVEPATH))

    return trainingData, trainingLabels, validationData, validationLabels


def getTrainingBatch(trainingData, trainingLabels, batchSize):
    temp = list(zip(trainingData, trainingLabels))
    random.shuffle(temp)
    data, label = zip(*temp)
    data = np.array(data)
    label = np.array(label)
    return data[:batchSize], label[:batchSize]


splitData()
