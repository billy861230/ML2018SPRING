import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam
from sklearn.cluster import KMeans
from keras.models import load_model
import time

start = time.time()
encoder = load_model('encoder.h5')

x = np.load('visualization.npy')
x = x.astype('float32') / 255.
x = np.reshape(x, (len(x), -1))
encoded_imgs = encoder.predict(x)
'''
print(encoded_imgs.shape)
encoded_imgs = encoded_imgs.reshape(encoded_imgs.shape[0], -1)
kmeans = KMeans(n_clusters=2, random_state=0).fit(encoded_imgs)
classify = []
classify.append([])
classify.append([])
for i in range(10000):
    if kmeans.labels_[i] == 0:
        classify[0].append(encoded_imgs[i])
    else:
        classify[1].append(encoded_imgs[i])

x = classify[0]
for i in range(len(classify[1])):
    x.append(classify[1][i])
print("end classify")
x = np.array(x)
#x = dimension_reduction(x)
'''
x_embedded = TSNE(n_components=2).fit_transform(encoded_imgs)
print('end tsne')
plt.scatter(x_embedded[:5000, 0], x_embedded[:5000, 1], c='b', label='dataset A', s=0.2)
plt.scatter(x_embedded[5000:, 0], x_embedded[5000:, 1], c='r', label='dataset B', s=0.2)
plt.legend()
plt.savefig('tsne.png')
end = time.time()
elapsed = end - start
print(elapsed)
