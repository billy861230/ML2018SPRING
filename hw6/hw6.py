import pandas as pd
import numpy as np
import csv
'''
dataset = pd.read_csv(
    "train.csv",
    sep='\t',
    names="TrainDataID,UserID,MovieID,Rating".split(","))
dataset.UserID = dataset.UserID.astype('category').cat.codes.values
dataset.MovieID = dataset.MovieID.astype('category').cat.codes.values
from sklearn.model_selection import train_test_split
train, val = train_test_split(dataset, test_size=0.1) '''

import keras
from keras.layers import Embedding, Add
from keras.optimizers import Adam
from keras.utils.vis_utils import model_to_dot
f = open('train.csv', 'r').read()
f = list(f.split('\n'))
del f[0]
usr = []
movie = []

rate = []
for line in f:
    if len(line) < 3:
        continue
    line = list(line.split(','))
    usr.append([int(line[1])])
    movie.append([int(line[2])])
    rate.append(int(line[3]))
usr = np.array(usr)
movie = np.array(movie)
x_train = [usr, movie]

y_train = np.array(rate)



movie_input = keras.layers.Input(shape=[1], name='Item')
movie_embedding = keras.layers.Embedding(
    3953, 15, name='Movie-Embedding')(movie_input)
movie_vec = keras.layers.Flatten(name='FlattenMovies')(movie_embedding)
movie_bias = Embedding(3953, 1)(movie_input)
user_input = keras.layers.Input(shape=[1], name='User')
user_vec = keras.layers.Flatten(name='FlattenUsers')(keras.layers.Embedding(
    6041, 15, name='User-Embedding')(user_input))
usr_bias = Embedding(6041, 1)(user_input)
prod = keras.layers.merge([movie_vec, user_vec], mode='dot', name='DotProduct')


model = keras.Model([user_input, movie_input], prod)

embed = keras.Model(movie_input, movie_embedding)

model.compile(optimizer='adam', loss='mse')
model.summary()


model.fit(x_train, y_train, batch_size=256, epochs=16)
model.save('test.hdf5')
embed.save('embedding.hdf5')

f = open('test.csv', 'r').read()
f = list(f.split('\n'))
del f[0]
usr = []
movie = []

rate = []
for line in f:
    if len(line) < 3:
        continue
    line = list(line.split(','))
    usr.append([int(line[1])])
    movie.append([int(line[2])])
usr = np.array(usr)
movie = np.array(movie)
x_test = [usr, movie]

'''
dataset = pd.read_csv(
    "train.csv", sep='\t', names="TestDataID,UserID,MovieID".split(","))
dataset.UserID = dataset.UserID.astype('category').cat.codes.values
dataset.MovieID = dataset.MovieID.astype('category').cat.codes.values'''
pred = model.predict(x_test)

result = [['TestDataID', 'Rating']]
for i in range(len(pred)):
    a = [i + 1, pred[i][0]]
    if pred[i][0] < 0:
        a[1] = 0
    elif pred[i][0] > 5:
        a[1] = 5

    result.append(a)

cout = csv.writer(open('result.csv', 'w'))
cout.writerows(result)