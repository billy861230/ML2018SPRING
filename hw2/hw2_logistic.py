import numpy as np
import pandas as pd
import csv
import math
import pickle
import sys


def sigmoid(x):
    y = [0] * len(x)
    for i in range(len(x)):
        z = x[i]
        y[i] = 1 / (1 + math.exp(-z))
    return np.array(y).astype('float64')


with open('logistic.pickle', 'rb') as file:
    w = pickle.load(file)

with open('b_logic.pickle', 'rb') as file:
    b = pickle.load(file)

t = open(sys.argv[1], "r")
string = t.read()
array = list(string.split())
del array[0]
test = []
for i in range(len(array)):
    single = list(map(float, array[i].split(',')))
    test.append(single)
t.close()

test = np.array(test)

test = np.matmul(test, np.transpose(w)) + b

result = [["id", "label"]]

result = [["id", "label"]]
i = 0

for i in range(len(test)):
    a = [i + 1]
    if test[i] > 0.5:
        a.append(1)
    else:
        a.append(0)
    result.append(a)
    i += 1
cout = csv.writer(open(sys.argv[2], 'w'))
cout.writerows(result)