import numpy as np

arr = np.fromfile('featureInfoHSV/paper_GH010386020.npy')
arr2 = np.fromfile('featureInfoHSV/shells_GH010349003.npy')

print(arr.size)
print(arr.shape)

print(arr2.size)
print(arr2.shape)

pcarr = np.fromfile('featureInfoPCA/pca_data.npy')
print(pcarr.shape)
print(pcarr.size)
