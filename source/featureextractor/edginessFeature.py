import cv2 as cv
from PIL import Image
import numpy as np

# returns a histogram of the lengths of the edges in an image
def extractEdges(image):

    edgeslist = []

    # iterates through the 90 different angles the image could be rotated at
    for i in range (0,90):
        
        rot_img = Image.fromarray(image).rotate(i)  # rotates the image by the degree
        rot_img_arr = np.uint8(np.array(rot_img.getdata()).reshape(image.shape)) # converts the image back to a numpy array
        cannyimg = cv.Canny(rot_img_arr, 100, 200) / 255  # gets edge outline of image using Canny algorithm and scales values to either 0 or 1
        cannyimg = thickerEdges(cannyimg)  # thickens edges in edge outline using thickerEdges function

        # iterates through rows/columns of image
        for k in range(0, image.shape[0] - 1):

            countv = 0
            counth = 0

            #iterates through columns/rows of image
            for j in range(0,image.shape[1] - 1):
                
                # finds vertical edges
                if cannyimg[j][k] != 0 or cannyimg[j][k+1] != 0 or cannyimg[j][k-1] != 0:
                    countv += 1
                else:
                    if countv > 20:
                        edgeslist.append(countv)
                    countv = 0

                # finds horizontal edges
                if cannyimg[k][j] != 0 or cannyimg[k+1][j] != 0 or cannyimg[k-1][j] != 0:
                    counth += 1
                else:
                    if counth > 20:
                        edgeslist.append(counth)
                    counth = 0

    edgeslist = np.array(edgeslist)
    histbins = [20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200]
    edgeshist, bins = np.histogram(edgeslist, bins=histbins)  # creates histogram of edge lengths from the list
    
    return edgeshist

# thickens the edges in an edge outline of an image
def thickerEdges(cannyimg):

    thickercanny = np.zeros(cannyimg.shape)
    
    #iterates through the image and make all pixels neighboring an edge also part of the edge
    for i in range (1, cannyimg.shape[0]-1):
        
        for j in range (1, cannyimg.shape[1]-1):
            if cannyimg[i-1][j-1] == 1 or cannyimg[i-1][j] == 1 or cannyimg[i-1][j+1] == 1 or cannyimg[i][j-1] == 1 or cannyimg[i][j] == 1 or cannyimg[i][j+1] == 1 or cannyimg[i+1][j-1] == 1 or cannyimg[i+1][j] == 1 or cannyimg[i+1][j+1] == 1:
                thickercanny[i][j] = 1
    
    return thickercanny


#def getVertEdges(cannyimg):
#    vedgelist = []
#    for i in range (50, 526):
#        count = 0
#        for j in range (50, 526):
#            if cannyimg[j][i] != 0 or cannyimg[j][i+1] != 0 or cannyimg[j][i-1] != 0:
#                    count += 1
#            else:
#                if count > 30:
#                     vedgelist.append(count)
#                count = 0
#    return vedgelist


#def getHorEdges(cannyimg):
#    hedgelist = []
#    for i in range (50, 526):
#        count = 0
#        for j in range (50, 526):
#            if cannyimg[i][j] != 0 or cannyimg[i+1][j] != 0 or cannyimg[i-1][j] != 0:
#                    count += 1
#            else:
#                if count > 30:
#                     hedgelist.append(count)
#                count = 0
#    return hedgelist


#def getAllEdges(image):
#    edgeslist = []
#    #     cannyimage = cv.Canny(image, 100, 200)
#    #     Image.fromarray(cannyimage).show()
#    for i in range(0, 90):
#        if i % 10 == 0:
#            print(i)
#        rot_img = Image.fromarray(image).rotate(i)
#        rot_img_arr = np.uint8(np.array(rot_img.getdata()).reshape(576, 576, 3))
#        cannyimg = cv.Canny(rot_img_arr, 100, 200) / 255
#        thickcanny = thickerEdges(cannyimg)
#        edgeslist.extend(getVertEdges(thickcanny))
#        edgeslist.extend(getHorEdges(thickcanny))
#
#    edgeslist = np.array(edgeslist)
#    histbins = [20,40,60,80,100,120,140,160,180,200,220,240,260,280,300]
#    edgeshist, bins = np.histogram(edgeslist, bins=histbins)
#    return edgeshist
