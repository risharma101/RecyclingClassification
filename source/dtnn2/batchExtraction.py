import numpy as np
import os
import random
import feature
import argparse
import sys

#parser = argparse.ArgumentParser()
#parser.add_argument('--pcaFileName', type=str, required=True)
#args = parser.parse_args()
#fileName = args.pcaFileName

#PCADATAPATH = '../featureextractor/featureInfoPCA/' + fileName

#if not os.path.exists('splitData'):
 #   os.makedirs('splitData')


#SAVEPATH = 'splitData/split_' + fileName.split(".")[0]

#if not os.path.exists(SAVEPATH):
 #   os.makedirs(SAVEPATH)



def splitData():
    
    if not len(sys.argv) == 2:
        print('ERROR! Wrong arguments to bathExtraction.py. Expecting: python batchExtraction.py [pca_File_Name]')
        sys.exit()
    elif not os.path.exists('../featureextractor/featureInfoPCA/'+ sys.argv[1].split(".")[0] + '.npy'):
        print('ERROR! Wrong arguments to bathExtraction.py. Expecting: python batchExtraction.py [pca_File_Name]')
        sys.exit()


    pcaFileName = sys.argv[1].split(".")[0] + '.npy'
    PCADATAPATH = '../featureextractor/featureInfoPCA/' + pcaFileName

    SAVEPATH = 'splitData/split_' + pcaFileName.split(".")[0]

    if not os.path.exists(SAVEPATH):
            os.makedirs(SAVEPATH)

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
    np.save('{0}/validation_labels.npy'.format(SAVEPATH), np.array(validationLabels))

def loadData(loadFolderName):
    loadPath = 'splitData/' + loadFolderName.split(".")[0]
    trainingData = np.load('./{0}/training_data.npy'.format(loadPath))
    trainingLabels = np.load('./{0}/training_labels.npy'.format(loadPath))
    validationData = np.load('./{0}/validation_data.npy'.format(loadPath))
    validationLabels = np.load('./{0}/validation_labels.npy'.format(loadPath))

    return trainingData, trainingLabels, validationData, validationLabels


def getTrainingBatch(trainingData, trainingLabels, batchSize):
    temp = list(zip(trainingData, trainingLabels))
    random.shuffle(temp)
    data, label = zip(*temp)
    data = np.array(data)
    label = np.array(label)
    return data[:batchSize], label[:batchSize]


splitData()
