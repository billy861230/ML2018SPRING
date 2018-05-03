import numpy as np
import pandas as pd
train_num = 140000

x = np.load('image.npy')
x = x.astype('float32') / 255.
x = np.reshape(x, (len(x), -1))
print(x.shape)
x_train = x[:train_num]
x_val = x[train_num:]
x_train.shape, x_val.shape

from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam

input_img = Input(shape=(784,))
encoded = Dense(256, activation='relu')(input_img)
encoded = Dense(128, activation='relu')(encoded)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)
encoded = Dense(16, activation='relu')(encoded)

decoded = Dense(32, activation='relu')(encoded)
decoded = Dense(64, activation='relu')(decoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(256, activation='relu')(decoded)
decoded = Dense(784, activation='sigmoid')(decoded)

encoder = Model(input=input_img, output=encoded)

adam = Adam(lr=5e-4)
autoencoder = Model(input=input_img, output=decoded)
autoencoder.compile(optimizer=adam, loss='mse')
autoencoder.summary()
autoencoder.fit(x_train, x_train, epochs=128, batch_size=256, shuffle=True)
#autoencoder.fit(x_train, x_train, epochs=128, batch_size=256, shuffle=True, validation_data=(x_val, x_val))
autoencoder.save('autoencoder.h5')
encoder.save('encoder.h5')

from sklearn.cluster import KMeans
encoded_imgs = encoder.predict(x)
encoded_imgs = encoded_imgs.reshape(encoded_imgs.shape[0], -1)
kmeans = KMeans(n_clusters=2, random_state=0).fit(encoded_imgs)
f = pd.read_csv('test_case.csv')
IDs, idx1, idx2 = np.array(f['ID']), np.array(f['image1_index']), np.array(f['image2_index'])
o = open('prediction.csv', 'w')
o.write("ID,Ans\n")
for idx, i1, i2 in zip(IDs, idx1, idx2):
    p1 = kmeans.labels_[i1]
    p2 = kmeans.labels_[i2]
    if p1 == p2:
        pred = 1
    else:
        pred = 0
    o.write("{},{}\n".format(idx, pred))
o.close()
