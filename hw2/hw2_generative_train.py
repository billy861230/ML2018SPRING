import numpy as np
import pandas as pd
import csv
import math
import pickle

def sigmoid(x):
    y = [0] * len(x)
    for i in range(len(x)):
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

train_x = np.array(train_x)
train_y = np.array(train_y)

train_x = train_x.astype('float64')
train_y = train_y.astype('float64')

t = open('test_X', 'r')
string = t.read()
array = list(string.split())
del array[0]
test = []
for i in range(len(array)):
    single = list(map(float, array[i].split(',')))
    test.append(single)
t.close()

test = np.array(test)

mu1 = np.zeros((123,))
mu2 = np.zeros((123,))

cnt1 = 0
cnt2 = 0

class1 = []
class2 = []


for i in range(len(train_y)):
    if train_y[i] == 0:
        mu1 += train_x[i]
        class1.append(i)
    else:
        mu2 += train_x[i]
        class2.append(i)

class1 = np.array(class1)
class2 = np.array(class2)
cnt1 = len(class1)
cnt2 = len(class2)
mu1 /= cnt1
mu2 /= cnt2
cov1 = np.cov(train_x[class1].T)
cov2 = np.cov(train_x[class2].T)
cov_inv = np.linalg.pinv((cnt1 * cov1 + cnt2 * cov2) / (cnt1 + cnt2))
w = np.matmul((mu1 - mu2).T, cov_inv)
b = -0.5 * np.matmul(np.matmul(mu1.T, cov_inv), mu1) + 0.5 * np.matmul(np.matmul(mu2.T, cov_inv), mu2) + np.log(cnt1 / cnt2)
y = 1 - (sigmoid(np.matmul(test, w) + b) > 0.5).astype(np.int)


file = open('generative.pickle', 'wb')
pickle.dump(w, file)
file.close()
file = open('b.pickle', 'wb')
pickle.dump(b, file)
file.close()


result = [["id", "label"]]
i = 0

for i in range(len(test)):
    a = [i + 1]
    a.append(y[i])
    result.append(a)
    i += 1
cout = csv.writer(open('generative.csv', 'w'))
cout.writerows(result)
