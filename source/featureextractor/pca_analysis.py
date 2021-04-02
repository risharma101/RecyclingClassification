import numpy as np
import utils

data = np.load('./featureInfo/paper_GH010377image-047.npy')
print(data[0])
#def joinVectors()

blobLen = len(data)
data = data[:,1:]
labels = np.tile(2, blobLen)
#labelData = np.hstack((np.expand_dims(labels, axis=1), data))


pcaData, pca = utils.getPCA(data)

