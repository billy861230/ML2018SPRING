import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from keras.models import load_model
encoder = load_model('encoder.h5')

x = np.load('image.npy')
x = x.astype('float32') / 255.
x = np.reshape(x, (len(x), -1))
encoded_imgs = encoder.predict(x)
encoded_imgs = encoded_imgs.reshape(encoded_imgs.shape[0], -1)
kmeans = KMeans(n_clusters=2, random_state=0).fit(encoded_imgs)
f = pd.read_csv('test_case.csv')
IDs, idx1, idx2 = np.array(f['ID']), np.array(f['image1_index']), np.array(f['image2_index'])
o = open('prediction.csv', 'w')
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
