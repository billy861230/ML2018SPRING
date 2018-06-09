import pandas as pd
import numpy as np
import csv
import sys
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
from keras.models import load_model

model = load_model('mf.hdf5')
f = open(sys.argv[1], 'r').read()
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

cout = csv.writer(open(sys.argv[2], 'w'))
cout.writerows(result)