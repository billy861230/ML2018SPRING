import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam
from sklearn.cluster import KMeans
from keras.models import load_model
import time
list_1 = []
list_2 = []
genres = []


table = pd.read_csv('movies.csv', sep='::')
print(table['Genres'])
for genre in table['Genres']:
    genres.append(genre)
for i in range(len(genres)):
    genres[i] = list(genres[i].split('|'))
    for genre in genres[i]:
        if ("Children\'s" in genre) or ("Adventure" in genre):
            list_1.append(i + 1)

        elif ("Musical" in genre) or ("Drama" in genre):
            list_2.append(i + 1)



start = time.time()
encoder = load_model('embedding.hdf5')
'''
f = open('movies.csv')
string = f.read()
f.close()
f = list(string.spilt())
del(f[0])
for line in f:
    line = list(line.spilt('::'))
    genre = list(line[2].spilt('|'))
    if ("Children's" in genre) or ("Adventure" in genre):
        list_1.append(int(line[0]))
    elif ("Musical" in genre) or ("Drama" in genre):
        list_2.append(int(line[0]))'''

x = []

for i in range(3952):
    x.append(i + 1)

x = np.array(x)


encoded_imgs = encoder.predict(x)
print(encoded_imgs)


encoded_imgs = encoded_imgs.reshape([3952, 13])
print(encoded_imgs.shape)
'''
encoded_imgs = encoded_imgs.reshape(encoded_imgs.shape[0], -1)
kmeans = KMeans(n_clusters=2, random_state=0).fit(encoded_imgs)
classify = []
classify.append([])
classify.append([])
for i in list_1:
    classify[0].append(i)
for i in list_2:
    classify[1].append(i)

x = classify[0]
for i in range(len(classify[1])):
    x.append(classify[1][i])
print("end classify")
x = np.array(x)
#x = dimension_reduction(x)'''

x_embedded = TSNE(n_components = 2).fit_transform(encoded_imgs)
x_1 = []
y_1 = []
for i in list_1:
    x_1.append(x_embedded[i][0])
    y_1.append(x_embedded[i][1])
x_2 = []
y_2 = []
for i in list_2:
    x_2.append(x_embedded[i][0])
    y_2.append(x_embedded[i][1])
x_1 = np.array(x_1)
y_1 = np.array(y_1)
x_2 = np.array(x_2)
y_2 = np.array(y_2)
print('end tsne')
plt.scatter(
    x_1,
    y_1,
    c='b',
    s=0.2)
plt.scatter(
    x_2,
    y_2,
    c='r',
    s=0.2)
plt.legend()
plt.savefig('tsne.png')
end = time.time()
elapsed = end - start
print(elapsed)
