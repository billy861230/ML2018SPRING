from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
import numpy as np
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
import sys

train = open(sys.argv[1], 'r')
string = train.read()
array = list(string.split('\n'))
del array[0]
train_x = []
train_y = []

input_shape = (48, 48, 1)
datagen = ImageDataGenerator(zca_whitening=False, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')


model = Sequential([
    BatchNormalization(input_shape=input_shape),
    Conv2D(63, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(63, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    BatchNormalization(),
    Conv2D(127, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(127, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    BatchNormalization(),
    Conv2D(255, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(255, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(255, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    BatchNormalization(),
    Conv2D(511, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(511, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(511, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    BatchNormalization(),
    Conv2D(511, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(511, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(511, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    BatchNormalization(),
    Flatten(),
    BatchNormalization(),
    Dropout(0.25),
    Dense(512, activation='relu'),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dense(7, activation='softmax')
])

model.compile(loss='categorical_crossentropy',  
              optimizer='adadelta',  
              metrics = ['accuracy'])

#print(model.summary())

for i in range(len(array) - 1):
    single = list(array[i].split(','))
    train_y.append(int(single[0]))
    single = list(map(float, single[1].split()))
    '''single = np.array(single)
    single.reshape(48, 48)'''
    train_x.append(single)

train_x = np.array(train_x)
train_y = np.array(train_y)



train_y = np_utils.to_categorical(train_y, 7)
train_x = train_x.reshape((28709, 48, 48, 1))
datagen.fit(train_x)


              
''', validation_split=0.2'''
model.fit_generator(datagen.flow(train_x, train_y, batch_size=16), steps_per_epoch=len(train_x), epochs=50)


model.save('my_model_vgg_best.hdf5')