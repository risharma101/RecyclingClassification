import numpy as np
import os

TRAININGFEATUREPATH = "../featureextractor/featureInfo/training/"
VALIDATIONFEATUREPATH = "../featureextractor/featureInfo/validation/"

# Get the training batch
# def getTrainingBatch(batchSize):


# Get the validation batch
def getTestingBatch(batchSize=None):
    testingPath = os.listdir(VALIDATIONFEATUREPATH)
    data = []
    labels = []
    for featureVector in testingPath:
        # The first column of all these is the label so we can extract that
        loadFeatureVector = np.load("{0}{1}".format(VALIDATIONFEATUREPATH, featureVector))
        featureLabel = [x[0] for x in loadFeatureVector]
        featureData = loadFeatureVector[:, 1:]
        data.append(featureData)
        labels.append(featureLabel)
    
    return np.array(data), np.array(labels)