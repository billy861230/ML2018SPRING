from skimage import io
import numpy as np
import sys
imgs = []

for i in range(415):
    img = io.imread('Aberdeen' + str(i) + '.jpg')
    img = np.array(img)
    print(img.shape())
    img = img.flatten()
    imgs.append(img)

imgs = np.array(imgs)
x = imgs.T
x_mean = np.mean(x, axis=1)

u, s, v=np.linalg.svd(x - x_mean, full_matrices=False)

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


