import pca_analysis
import utils
import random
import numpy as np

dataArr, labelArr = pca_analysis.joinVectors()

def handleLDA(dataArr, labelArr):
    temp = list(zip(dataArr, labelArr))
    random.shuffle(temp)
    dataArr, labelArr = zip(*temp)
    x_train, y_train, x_test, y_test = utils.getLDA(dataArr, labelArr)
    #print(ldaData.shape)
    np.save('featureInfoPCA/lda_x_train.npy', x_train)
    np.save('featureInfoPCA/lda_y_train.npy', y_train)
    np.save('featureInfoPCA/lda_x_test.npy', x_test)
    np.save('featureInfoPCA/lda_y_test.npy', y_test)
    ldaData = np.concatenate(x_train, x_test)
    ldaLabels = np.concatenate(y_train, y_test)
    fullData = np.hstack((np.expand_dims(ldaLabels, axis=1), ldaData))

    np.save('featureInfoPCA/lda_data.npy', fullData)

handleLDA(dataArr, labelArr)
