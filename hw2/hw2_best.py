from sklearn import datasets
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import csv
import sys


def sigmoid(x):
    y = [0] * len(x)
    for i in range(len(x)):
        z = x[i]
        y[i] = 1 / (1 + math.exp(-z))
    return np.array(y).astype('float64')


x = open(sys.argv[1], "r")
string = x.read()
array = list(string.split())
del array[0]
train_x = []
for i in range(32561):
    single = list(map(float, array[i].split(',')))
    #del single[10]
    train_x.append(single)
x.close()
y = open(sys.argv[2], 'r')
string = y.read()
train_y = list(string.split())
y.close()

train_x = np.array(train_x)
train_y = np.array(train_y)


train_x = train_x.astype('float64')
train_y = train_y.astype('float64')

t = open(sys.argv[3], 'r')
string = t.read()
array = list(string.split())
del array[0]
test = []
for i in range(len(array)):
    single = list(map(float, array[i].split(',')))
    #del single[10]
    test.append(single)
t.close()

test = np.array(test)

sc = StandardScaler()
sc.fit(train_x)
train_x_std = sc.transform(train_x)
sc.fit(test)
test_std = sc.transform(test)

lr = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
#lr = LogisticRegression(penalty='l1')
lr.fit(train_x_std, train_y)

predict = lr.predict_proba(test_std)
print(predict)
check = lr.predict_proba(train_x_std)

result = [["id", "label"]]
'''
string = ""
for i in range(len(train_x)):
    if check[i][0] > check[i][1]:
        string += '0'
    else:
        string += '1'
    string += '\n'

file = open('test', 'w')
file.write(string)
file.close()
'''

for i in range(len(test)):
    a = [i + 1]
    if predict[i][0] > predict[i][1]:
        a.append(0)
    else:
        a.append(1)
    result.append(a)
    i += 1
cout = csv.writer(open(sys.argv[4], 'w'))
cout.writerows(result)
