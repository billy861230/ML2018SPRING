import numpy as np
np.random.seed(1337) # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
import sys
from keras.models import load_model
import csv

model = load_model('my_model.hdf5')
#print(model.summary())

test = open(sys.argv[1], 'r')
string = test.read()
array = list(string.split('\n'))
del array[0]
test_x = []
'''
single = list(array[0].split(','))
print(single)
single = list(map(float, single[1].split()))
print(len(single))

'''

for i in range(len(array) - 1):
    single = list(array[i].split(','))
    #print(single[0])
    single = list(map(float, single[1].split()))
    '''single = np.array(single)
    single.reshape(48, 48)'''
    test_x.append(single)

test_x = np.array(test_x)

test_x = test_x.reshape((len(array) - 1, 48, 48, 1))

predict = model.predict(test_x)
print(len(predict))

result = [["id", "label"]]


for i in range(len(predict)):
    a = [i]
    b = 0
    for j in range(7):
        if predict[i][j] > predict[i][b]:
            b = j
    a.append(b)
    #a.append(predict[i])
    result.append(a)
    i += 1
cout = csv.writer(open(sys.argv[2], 'w'))
cout.writerows(result)