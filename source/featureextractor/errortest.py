import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import cv2
import utils
import edginessFeature as edginess
import edgyFeat as edgyness

#hsvList = os.listdir('featureInfoHSV/')
#edgeList = os.listdir('featureInfoEdge/')
#hsv1 = np.load('featureInfoHSV/' + hsvList[0])
#hsv2 = np.load('featureInfoHSV/' + hsvList[1])
#edge1 = np.load('featureInfoEdge/' + edgeList[0])
#edge2 = np.load('featureInfoEdge/' + edgeList[1])

#print('hsv1: ' , hsv1.shape)
#print('hsv2: ' , hsv2.shape)
#print('edge1: ' , edge1.shape)
#print('edge2: ' , edge2.shape)


#pcaList = os.listdir('featureInfoPCA/')
#for file in pcaList:
#    print(file, ': ', np.load('featureInfoPCA/' + file).shape)

#for file in hsvList:
#    print(file, np.load('featureInfoHSV/' + file).shape)

#print(len(hsvList))
#print(len(edgeList))


imgPath = '/home/mlguest1/MetroRecyclingClassification/source/PMSImages/cardboard/'
imgList = os.listdir(imgPath)
img = cv2.imread(imgPath + imgList[0])
blobs, markers, labels = utils.extractBlobs(img)
edgehist = edginess.extractEdges(blobs[0])
for b in blobs:
    edgehist = edginess.extractEdges(b)
    print(edgehist)
    edgyhist = edgyness.extractEdges(b)
    print(edgyhist)
