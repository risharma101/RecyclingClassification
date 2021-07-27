import numpy as np
import os
import utils
import argparse ###

###
parser = argparse.ArgumentParser()
parser.add_argument('--dataFolderPath',type=str,required=True)
parser.add_argument('--saveFileName',type=str,required=True)
args = parser.parse_args()

folderPath = args.dataFolderPath 
###

#folderPath = 'featureInfoHSV/'

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
    
    path = 'featureInfoPCA/'
    if not os.path.exists(path):
        os.makedirs(path)

    saveFileName = args.saveFileName
    np.save(path + saveFileName, fullData)

handlePCA(dataArr, labelArr)

