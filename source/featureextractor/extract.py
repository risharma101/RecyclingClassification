#############################################################################################################
#Author: Masa Hu
#
#                               Trash Classification Project
#############################################################################################################
#NATIVE LIBRARY IMPORTS
import argparse
import time
import re
import os
import pickle
from multiprocessing import Process
from multiprocessing import Manager
import gc

#OPEN SOURCE IMPORTS
import numpy as np
import cv2

#CUSTOM IMPORTS
import gabor_threads_roi as gabor
import utils
import backgroundFeatureInfo as background
import mathCalculations as calculation
############################################################################################################
#ARGUMENT PARSER FOR FEATURES TO EXTRACT
parser = argparse.ArgumentParser()
parser.add_argument('--gabor',default=False,action='store_const',const=True)
parser.add_argument('--hog',default=False,action='store_const',const=True)
parser.add_argument('--rgb',default=False,action='store_const',const=True)
parser.add_argument('--hsv',default=False,action='store_const',const=True)
parser.add_argument('--size',default=False,action='store_const',const=True)
# parser.add_argument('--file',type=str,required=True)
parser.add_argument('--fileType', type=str,required=True)
parser.add_argument('--folderPath', type=str, required=True)
args = parser.parse_args()

#############################################################################################################
#FEATURE EXTRACTOR TO MULTIPROCESS
def extractFeatures(img,label,datadump):
    f1 = utils.getRGBHist(img)
    f2 = utils.getHSVHist(img)
    f3 = utils.getHOG(img)
    f4 = gabor.run_gabor(img,gabor.build_filters(16)).flatten()
    f5 = utils.getSize(img)
    datadump.append([np.hstack((f1,f2,f3,f4,f5)),label])

#HELPER FUNCTION TO GET LABEL BASED ON FILENAME
def getLabel(filename):
    group = re.findall("shells|plastic_bottles|plastic_bags|paper|metal_cans|cardboard|assorted",filename)
    if(len(group) == 0): return -1
    elif(group[0] == 'shells'): return 0
    elif(group[0] == 'plastic_bottles'): return 1
    elif(group[0] == 'plastic_bags'): return 2
    elif(group[0] == 'paper'): return 3
    elif(group[0] == 'metal_cans'): return 4
    elif(group[0] == 'cardboard'): return 5
    elif(group[0] == 'assorted'): return -1
    else: print('image belongs to no group'); return -1

#############################################################################################################
#############################################################################################################
if __name__ == '__main__':

    # FILENAME = args.file
    FOLDERPATH = args.folderPath
    FILETYPE = args.fileType
    # img = cv2.imread('testImages/' + FILENAME)
    label = getLabel(FILETYPE)

    imgFileList = os.listdir(FOLDERPATH)

    # We will just split up the features here and track whether they should be training/validation
    trainingLen = int(len(imgFileList)*.8)
    isTrainingImg = True
    counter = 0

    for imgName in imgFileList:
        counter += 1
        if counter > trainingLen:
            isTrainingImg = False
        
        img = cv2.imread("{0}{1}".format(FOLDERPATH,imgName))
        # The 'ground truths' for getting rid of background blobs
        rgbBackgroundFeatures = background.handleBackgroundBlobs()

        # We need to iterate over all the files in the dir (we will split it 80/20 for training vs test)

        #segment the image
        blobs, markers,labels = utils.extractBlobs(img)

        rgbFeatureList = []
        #multiprocess the feature extraction
        manager = Manager()
        datadump = manager.list()
        jobs = []
        max_process = 30
        for i,b in enumerate(blobs):
            # This is strictly for getting the rgb value to figure out which blobs are the background
            rgbFilter = utils.getRGBHist(b)
            rgbFeatureList.append(rgbFilter)

            p = Process(target=extractFeatures,args=(b,label,datadump))
            jobs.append(p)
            p.start()

            if i % max_process == max_process - 1:
                for j in jobs: j.join()
        for j in jobs: j.join()
        data = np.array([d[0] for d in datadump])
        label = np.array([d[1] for d in datadump])
        del datadump
        del jobs
        gc.collect()

        #Background calculations

        # Pairwise diff calculation
        # pairwiseDiff, minIndexList = calculation.calcPairwiseDiff(backgroundBlobs, rgbFeatureList)
        # background.graphBackgroundCalculation(pairwiseDiff, minIndexList, FILENAME.split(".")[0])

        #Cosine similarity calcualtion
        cosineSimilarity, cosineSimMinIndexList = calculation.calcCosineSimilarity(rgbBackgroundFeatures, rgbFeatureList)
        #background.graphBackgroundCalculation(cosineSimilarity, cosineSimMinIndexList, FILENAME.split(".")[0])

        #MSE calculation
        # mse, mseMinIndexList = calculation.calcMSE(backgroundBlobs, rgbFeatureList)
        # background.graphMSECalculation(mse, mseMinIndexList)

        # Delete background blobs (for now anything with cos similarity above .8 score)
        backgroundIndexes = [n for n in range(len(cosineSimilarity)) if cosineSimilarity[n] >= 0.8]
        # data = np.delete(data, backgroundIndexes, axis=0)


        #PCA ANALYSIS ON DATA INSTANCES. POSSIBLY NOT NECESSARY?
        # data,pca = utils.getPCA(data)
        data = np.hstack((np.expand_dims(label,axis=1),data))

        #SAVE OUTPUT
        if not os.path.exists('featureInfo'):
            os.makedirs('featureInfo')
        if not os.path.exists('featureInfo/training'):
            os.makedirs('featureInfo/training')
        if not os.path.exists('featureInfo/validation'):
            os.makedirs('featureInfo/validation')
        
        if isTrainingImg:
            np.save(os.path.join('featureInfo/training',"{0}_{1}".format(FILETYPE, imgName.split(".")[0])),data)
        else:
            np.save(os.path.join('featureInfo/validation',"{0}_{1}".format(FILETYPE, imgName.split(".")[0])),data)
