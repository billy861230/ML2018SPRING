import numpy as np
import pandas as pd
import csv
import math
import sys
import pickle


def sigmoid(x):
    y = [0] * len(x)
    for i in range(len(x)):
        '''if x[i] > 10:
            z = 10 ** (-6)
        elif x[i] < -10:
            z = -10
        else:'''
        z = x[i]
        y[i] = 1 / (1 + math.exp(-z))

    return np.array(y).astype('float64')


x = open('train_X', "r")
string = x.read()
array = list(string.split())
del array[0]
train_x = []
for i in range(32561):
    single = list(map(float, array[i].split(',')))
    train_x.append(single)
x.close()
y = open('train_Y', 'r')
string = y.read()
train_y = list(string.split())
y.close()

max = [0] * 123

for j in range(123):
    if train_x[0][j] == 0 or train_x[i][j] == 1:
        continue
    for i in range(len(train_x)):
        if max[j] < train_x[i][j]:
            max[j] = train_x[i][j]

for j in range(123):
    if train_x[0][j] == 0 or train_x[i][j] == 1:
        continue
    for i in range(len(train_x)):
        train_x[i][j] /= max[j]

iterator = 12000
w = []

for i in range(123):
    w.append(0)
b = 0

train_x = np.array(train_x).astype('float64')
train_y = np.array(train_y).astype('float64')
w = np.array(w).astype('float64')


l_rate = 0.0001

'''
z = np.matmul(train_x, np.transpose(w)) + b
print(z)
y = sigmoid(z)
y = y.astype('float64')

cross_entropy = np.matmul((1 - train_y), np.log(1 - y)) - np.matmul(train_y, np.log(y))

print(cross_entropy)
'''


for i in range(iterator):
    z = np.matmul(train_x, np.transpose(w)) + b
    print(z)
    y = sigmoid(z)
    y = y.astype('float64')
    cross_entropy = np.matmul((1 - train_y), np.log(1 - y)) - np.matmul(train_y, np.log(y))
    if abs(cross_entropy) < 1:
        break
    w_grad = - (1 / 36521) * np.matmul(np.transpose(train_x), (train_y - y))
    w -= (l_rate * w_grad)
    b_grad = np.mean(-1 * (train_y - y))
    b -= (l_rate * b_grad)
    print(cross_entropy)
    print(i)

file = open('logistic.pickle', 'wb')
pickle.dump(w, file)
file.close()
file = open('b_logic.pickle', 'wb')
pickle.dump(b, file)
file.close()

t = open('test_X', 'r')
string = t.read()
array = list(string.split())
del array[0]
test = []
for i in range(len(array)):
    single = list(map(float, array[i].split(',')))
    test.append(single)
t.close()

test = np.matmul(test, np.transpose(w)) + b

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
cout = csv.writer(open('ans1.csv', 'w'))
cout.writerows(result)
