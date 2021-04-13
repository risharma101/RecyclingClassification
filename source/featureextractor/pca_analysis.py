import numpy as np
import os
import utils

folderPath = 'featureInfoHSV/'

featureVectorList = os.listdir(folderPath)

# Join the feature vectors all together into one array so pca can be run as whole on them
def joinVectors():
    allData = []
    allLabels = []
    for imgFeatures in featureVectorList:
        if imgFeatures.endswith('.swp'):
            continue
        featureVectors = np.load('./{0}/{1}'.format(folderPath, imgFeatures))
        data = featureVectors[:, 1:] # Get rid of None in first column
        
        currLabel = utils.getLabel(imgFeatures)
        labelArr = np.tile(currLabel, len(data))

        if allData == []:
            allData = data
            allLabels = labelArr
        else:
            allData = np.concatenate((allData, data), 0)
            allLabels = np.concatenate((allLabels, labelArr), 0)

        print(allLabels.shape)
        print(allData.shape)

    return allData, allLabels


dataArr, labelArr = joinVectors()

def handlePCA(dataArr, labelArr):
    pcaData, pca = utils.getPCA(dataArr)
    print(pcaData.shape)
    fullData = np.hstack((np.expand_dims(labelArr, axis=1), pcaData))

    np.save('featureInfoPCA/pca_data_hsv_rgb.npy', fullData)

handlePCA(dataArr, labelArr)

