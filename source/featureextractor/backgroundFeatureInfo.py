import numpy as np
import utils
import gabor_threads_roi as gabor
import random
import cv2
import csv
import sys
import matplotlib.pyplot as plt
import pymeanshift as pms
import os

np.set_printoptions(threshold=sys.maxsize)

# We just want to get the background blob information
def getBackgroundBlobInfo(img, markers, ignoreLabel):
    blobs = []
    labels = np.unique(markers)
    img = cv2.imread(img)
    canvas = img.copy()

    for uq_mark in labels:
        #get the segment and append it to inputs
        if (uq_mark == ignoreLabel):
            region = img.copy()
            region[markers != uq_mark] = [0,0,0]
            #opencv bounding rect only works on single channel images...
            blank = img.copy()
            blank = blank - blank
            blank[markers == uq_mark] = [255,255,255]
            grey = cv2.cvtColor(blank,cv2.COLOR_BGR2GRAY)
            x,y,w,h = cv2.boundingRect(grey)
            cropped = region[y:y+h,x:x+w]
            cropped = np.uint8(cropped)
            blob = cv2.resize(cropped,(64,64),interpolation=cv2.INTER_LINEAR)
            blobs.append(blob)
            b = random.randint(0,255)
            g = random.randint(0,255)
            r = random.randint(0,255)
            canvas[markers == uq_mark] = [b,g,r]

    return blobs

# Run the feature extraction on all the background image in order to do comparison
# We only need to pay attention to the RGB feature as that is what pairwise diff is calc on
def getBackgroundFeatureInfo():
    backgroundFeatureList = []
    filePathExtension = './backgroundChecker/'
    data = np.genfromtxt(filePathExtension + 'backgroundExclusionLabel.csv', delimiter=',', skip_header=1, dtype=None, encoding='utf8')
    # array of tuples of form (imageName, backgroundLabel)
    for x in range(len(data)):
        featureInfo = []
        fileName = data[x][0].split('.')[0]
        markers = np.genfromtxt(filePathExtension + fileName + '.txt', delimiter=',')
        
        backgroundBlob = getBackgroundBlobInfo(filePathExtension + data[x][0], markers, data[x][1])

        #print('Background blob: ', np.array(backgroundBlob).shape)
        
        # Should be list of 1
        for i, b in enumerate(backgroundBlob):
            rgbData = utils.getRGBHist(b)
            featureInfo.append(rgbData)

        backgroundFeatureList.append(np.array(featureInfo[0]))

    return np.array(backgroundFeatureList)


# This is for determining the background, show be run through by 5 images and pick out the blobs that are the background by hand
def determineBackground(original, imageFileName, labelFileName, SHOW=True, SPATIAL_RADIUS=5,RANGE_RADIUS=5,MIN_DENSITY=250):
    segmented_image,labels_image,number_regions = pms.segment(
            original,
            spatial_radius=SPATIAL_RADIUS,
            range_radius=RANGE_RADIUS,
            min_density=MIN_DENSITY)
    
    print("Number of Regions Found: %s" % number_regions)
    unique_labels = np.unique(labels_image)

    # Save the image and also save the labels to a folder
    if SHOW:
        if not os.path.exists('backgroundChecker'):
            os.makedirs('backgroundChecker')
        cv2.imwrite(os.path.join('backgroundChecker', imageFileName), segmented_image)
        np.savetxt(os.path.join('backgroundChecker', labelFileName), labels_image, delimiter=',')


def handleBackgroundBlobs():
    filePathExtension = './backgroundImages/'
    # labels of the background images
    data = np.genfromtxt(filePathExtension + 'backgroundExclusionLabel.csv', delimiter=',', skip_header=1, dtype=None, encoding='utf8')
    rgbFilteredBackground = [] # Will not be length 4 of background images
    for x in range(len(data)):
        backgroundLabel = data[x][1]
        fileName = filePathExtension + data[x][0]
        blobs, markers, labels = utils.extractBlobs(cv2.imread(fileName), findBackground=True, ignoredBlob=backgroundLabel)
        # Should just be one
        for i,b in enumerate(blobs):
            rgbBackground = utils.getRGBHist(b)
            rgbFilteredBackground.append(rgbBackground)
    return rgbFilteredBackground

# Will take in a 1D array of the 
def graphBackgroundCalculation(arr, index, outputFile=None, csvOutput=False):
    if csvOutput:
        csvFile = open('tester.csv', 'w')
        for i,j in zip(arr, index):
            csvFile.write(str(i)+"," + str(j))
        csvFile.close()

    plt.figure(figsize=(22,14))
    #plt.plot(np.sort(index), np.sort(arr), color="red")
    plt.plot(index, arr, color="red")
    plt.xticks(range(len(index)),index, rotation=90)
    if outputFile != None:
        plt.savefig(os.path.join('backgroundSimilarityGraph', outputFile + '_bargraph.png'))
    plt.show()
