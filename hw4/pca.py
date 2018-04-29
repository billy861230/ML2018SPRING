from skimage import io
import numpy as np
import sys
imgs = []

for i in range(415):
    img = io.imread('Aberdeen/' + str(i) + '.jpg')
    img = np.array(img)
    img = img.flatten()
    imgs.append(img)

imgs = np.array(imgs)

x_mean = np.mean(imgs, axis=0)
x = imgs.T
mean = []
for i in range(415):
    mean.append(x_mean)
mean = np.array(mean)

u, s, v=np.linalg.svd(x - mean.T, full_matrices=False)

y = list(sys.argv[1].split('.'))
y = int(y[0])
y = imgs[y].T
y = y.T
y = y - x_mean

k = []
u = u.T

for i in range(4):
    w = np.dot(y, u[i])
    k.append(w)

k = np.array(k)
y = np.dot(k, u[0:4])
x_mean = x_mean.T
y = y + x_mean
y = y.reshape(600, 600, 3)