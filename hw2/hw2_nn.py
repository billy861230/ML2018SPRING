import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam
import csv
from keras.models import load_model
from keras.utils import np_utils
import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats


x = open('train_X', "r")
string = x.read()
array = list(string.split())
del array[0]
train_x = []
for i in range(32561):
    single = list(map(float, array[i].split(',')))
    del single[10]
    train_x.append(single)
x.close()
y = open('train_Y', 'r')
string = y.read()
train_y = list(map(float, string.split()))
y.close()
train = train_x
label = train_y

train_x = np.matrix(train_x)
train_y = np.matrix(train_y).T

#print(train_y)

t = open('test_X', 'r')
string = t.read()
array = list(string.split())
del array[0]
test = []
for i in range(len(array)):
    single = list(map(float, array[i].split(',')))
    del single[10]
    test.append(single)
t.close()
test = np.array(test)

#end load data
#data normalized
sc = StandardScaler()
sc.fit(train_x)
train_x_std = sc.transform(train_x)

#print(train_x_std)
test_std = sc.transform(test)

cov = []
a = 0
b = 0

print(train[157][42])
for i in range(len(train_y)):
    if train_y[i][0] == 1:
        continue
    b += 1
    
    if train[i][42] == 0:
        a += 1
        
print(b/len(train_y))


model = Sequential()
model.add(Dense(32, activation='relu', input_dim=122))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics = ['accuracy'])


'''
def build_model():
    model = Sequential()
    model.add(Dense(input_dim=122, units=32, activation='relu'))
    model.add(Dense(units=32, activation='sigmoid'))
    model.add(Dense(units=32, activation='sigmoid'))
    model.add(Dense(units=1, activation='softmax'))
    model.summary()
    return model

model = build_model()
model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
#end model construct'''

model.fit(train_x_std, train_y, epochs=10, batch_size=32)
test = sc.transform(test)


score = model.evaluate(train_x_std, train_y)
print('\nTrain Acc:', score[1])
predict = model.predict_classes(test)
print(predict)


result = [["id", "label"]]

for i in range(len(test)):
    a = [i + 1]
    a.append(predict[i][0])
    result.append(a)
    i += 1
cout = csv.writer(open('nn1.csv', 'w'))
cout.writerows(result)

model.save('my_model.h5')