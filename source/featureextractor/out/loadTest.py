import numpy as np

with open('test.npy', 'rb') as f:
    # The first entry in each row is the ground truth value
    print(np.load(f))
