import os
import cv2
from scipy.misc import face
from skimage import color, data, restoration
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft2, ifft2

#stores best angle and best strength
best_strength = None
best_angle = None
best_matrix = None

#place where images are stored
filepath = '/mnt/storage2/METRO_recycling/john_images_for_IIC'

def wiener_filter(img, kernel, K = 10):
    dummy = np.copy(img)
    kernel = np.pad(kernel, [(0, dummy.shape[0] - kernel.shape[0]), (0, dummy.shape[1] - kernel.shape[1])], 'constant')
    # Fourier Transform
    dummy = fft2(dummy)
    kernel = fft2(kernel)
    kernel =  np.conj(kernel) / (np.abs(kernel) ** 2 + K)
    dummy = dummy * kernel
    dummy = np.abs(ifft2(dummy))
    return np.uint8(dummy)

#goes through images in given directory
for image in os.listdir(filepath):
    print(image)

    #reads the image, converts it to grey, and measures the blur
    image = cv2.imread(filepath + "/" + image)
    gray = color.rgb2gray(image)
    best_blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    print("First blur: " + str(best_blur))
    #plt.imshow(gray)
    #plt.show()
    #goes through psf file and reads it line by line
    f = open("all_psf.txt")
    strength = None
    angle = None
    psf_matrix = None
    linecount = 0
    for line in f:
        
        linecount += 1
        if linecount < 0:
            continue

        if line[0] == "i":

            #if the matrix is not empty
            if not (strength == None and angle == None):
                #print("PSF matrix")
                #print(psf_matrix)

                #deconvolutes image using given filter and measures blur again
                #filtered_image = restoration.unsupervised_wiener(gray, psf_matrix.astype(float))
                filtered_image = wiener_filter(gray, psf_matrix)
                #print("Filtered")
                #print(filtered_image)
                #for i in range (0, 575):
                #    for j in range (0, 575):
                #        if filtered_image[0][i][j] == 1:
                #            filtered_image[0][i][j] = gray[i][j]

                new_blur = cv2.Laplacian(filtered_image[0], cv2.CV_64F).var()
               
                plt.imshow(filtered_image)
                plt.show()
                #if blur is better, record it
                if new_blur >0:
                    plt.subplot(121), plt.imshow(gray)
                    plt.title('Original Gray')
                    plt.subplot(122), plt.imshow(filtered_image[0])
                    plt.title('Deblurred: ' + str(strength) + ', ' + str(angle))
                    plt.show()
                    pass
                print("New blur: " + str(new_blur))
                if new_blur > best_blur:
                    best_blur = new_blur
                    best_strength = strength
                    best_angle = angle
                    best_matrix = psf_matrix
                psf_matrix = None
                
            #read the new strength
            strength = int(line.strip().split(" ")[-1])
            print("Testing strength: " + str(strength)) 
        elif line[0] == "j":
            #read the new angle
            angle = int(line.strip().split(" ")[-1])
            print("Testing angle: " + str(angle))
        else:
            #reads the psf matrix and stores it 
            if psf_matrix is  None:
                psf_matrix = np.array([line.strip().split("\t")])
            else:
                #print("current array")
                #print(psf_matrix)
                #print("New array")
                new_array = np.array(line.strip().split("\t"))
                #print(new_array)
                psf_matrix = np.vstack([psf_matrix, new_array])
    f.close()
    print(best_strength)
    print(best_angle)
    print(best_matrix)

    filtered_image = restoration.unsupervised_wiener(gray, best_matrix.astype(float))
   
    plt.subplot(121), plt.imshow(gray)
    plt.title('Original Gray')
    plt.subplot(122), plt.imshow(filtered_image[0])
    plt.title('Best Deblurred: ' + str(best_strength) + ', ' + str(best_angle))
    plt.show()

    break
