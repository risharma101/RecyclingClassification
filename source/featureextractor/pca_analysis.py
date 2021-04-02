import numpy as np
import os
import utils

folderPath = 'featureInfo/'

featureVectorList = os.listdir(folderPath)

# Join the feature vectors all together into one array so pca can be run as whole on them
def joinVectors():
    allData = []
    allLabels = []
    for imgFeatures in featureVectorList:
        featureVectors = np.load('./{0}/{1}'.format(folderPath, imgFeatures))
        data = featureVectors[:, 1:] # Get rid of None in first column

        currLabel = utils.getLabel(imgFeatures)
        labelArr = np.tile(currLabel, len(data))

        allData = np.append(allData, data)
        allLabels = np.append(allLabels, labelArr)

    print(allData.shape)

pcaData, pca = utils.getPCA(data)

