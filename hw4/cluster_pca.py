from skimage import io
import numpy as np
from skimage import transform
import sys
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
x = np.load(sys.argv[1])
x = x.astype('float32') / 255.
x = np.reshape(x, (len(x), -1))
print(x.shape)
x_mean = np.mean(x, axis=0)
x = x.T
mean = []
for i in range(140000):
    mean.append(x_mean)
mean = np.array(mean)

u, s, v = np.linalg.svd(x - mean.T, full_matrices=False)

v = v.T
kmeans = KMeans(n_clusters=2, random_state=0).fit(v)
f = pd.read_csv(sys.argv[2])
IDs, idx1, idx2 = np.array(f['ID']), np.array(
    f['image1_index']), np.array(f['image2_index'])
o = open(sys.argv[3], 'w')
o.write("ID,Ans\n")
for idx, i1, i2 in zip(IDs, idx1, idx2):
    p1 = kmeans.labels_[i1]
    p2 = kmeans.labels_[i2]
    if p1 == p2:
        pred = 1
    else:
        pred = 0
    o.write("{},{}\n".format(idx, pred))
o.close()

