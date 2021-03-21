import numpy as np

def extractBackgroundFeature():
    pairwiseDiff = []

    groundTruth = np.genfromtxt('groundTruthFeatures.csv', delimiter=',', dtype=None, encoding='utf8')
    featureVector = np.genfromtxt('featureVectorTest.csv', delimiter=',', dtype=None, encoding='utf8')
    # These are in (r,g,b) color space (256,256,256)
    # print(groundTruth.shape)
    # print(len(featureVector))

    # all arrays should be 768 for feature vector length
    # # Iterate over each blob of the new image
    for x in range(len(featureVector)):
        featurePairwiseDiff = []
        # iterate over the 5 ground truth blobs
        for i in range(len(groundTruth)):
            featurePairwiseDiff = np.append(featurePairwiseDiff, np.sum(np.subtract(groundTruth[i], featureVector[x])) / len(groundTruth[i]))

        # print("Blob val: " + str(np.sum(featurePairwiseDiff) / len(groundTruth)))        
        pairwiseDiff = np.append(pairwiseDiff, abs(np.sum(featurePairwiseDiff) / len(groundTruth)))

    minIndexList = np.argsort(pairwiseDiff)
    # pairwiseDiff = pairwiseDiff.sort()
    print(np.sort(pairwiseDiff))
    print(minIndexList)
    return pairwiseDiff.sort(), minIndexList
        
#extractBackgroundFeature()

import os
imgFilePath = os.listdir("../../images/cardboard/")
print(len(imgFilePath))
for imgName in imgFilePath:
    print(imgName)
