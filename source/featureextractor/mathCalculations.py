import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
import utils
'''
groundTruth will come in as (5, # of feature vectors for RGB)
feature vector will come in as (# of blobs, # of feature vectors for RGB)
The number of feature vectors will be equal for both parameters
'''
def calcPairwiseDiff(groundTruth, featureVector):
    pairwiseDiff = []

    for x in range(len(featureVector)):
        featurePairwiseDiff = []
        # iterate over the 5 ground truth blobs
        for i in range(len(groundTruth)):
            featurePairwiseDiff = np.append(featurePairwiseDiff, np.sum(np.subtract(groundTruth[i], featureVector[x])) / len(groundTruth[i]))

        pairwiseDiff = np.append(pairwiseDiff, abs(np.sum(featurePairwiseDiff) / len(groundTruth)))

    minIndexList = sorted(range(len(pairwiseDiff)), key=lambda x: pairwiseDiff[x])

    return pairwiseDiff, minIndexList

#Maximization function
def calcCosineSimilarity(groundTruth, featureVector):
    cosineSimilarity = []
    indexList = []
    #print(max(featureVector[0]))
    for x in range(len(featureVector)):
        #print(max(featureVector[x]))
        indexList = np.append(indexList, x)
        featureCosineSimilarity = []
        for i in range(len(groundTruth)):
            cosSim = cosine_similarity(featureVector[x].reshape(1,-1), groundTruth[i].reshape(1,-1))
            #if x == 0:
                #print(cosSim)
            featureCosineSimilarity = np.append(featureCosineSimilarity, cosSim)
        cosineSimilarity = np.append(cosineSimilarity, np.sum(featureCosineSimilarity) / len(groundTruth))
    
    #maxIndexList = sorted(range(len(cosineSimilarity)), key=lambda x: cosineSimilarity[x])[::-1]
    zippedData = zip(cosineSimilarity, indexList)
    zippedData.sort(key=lambda x: x[1])
    seperated = zip(*zippedData)
    similarity, blobNumber = list(seperated[0]), list(seperated[1])
    return similarity, blobNumber

# Minimization function
def calcMeanSquareError(groundTruth, featureVector):
    mseSimilarity = []
    for x in range(len(featureVector)):
        featureMSE = []
        for i in range(len(groundTruth)):
            MSE = mean_squared_error(featureVector[x].reshape(1,-1), groundTruth[i].reshape(1,-1))
            featureMSE = np.append(featureMSE, MSE)
        mseSimilarity = np.append(mseSimilarity, np.sum(featureMSE) / len(groundTruth))

    minIndexList = sorted(range(len(mseSimilarity)), key=lambda x: mseSimilarity[x])

    return mseSimilarity, minIndexList
