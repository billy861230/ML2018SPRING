from skimage import io
import numpy as np
from skimage import transform
import sys
imgs = []

for i in range(415):
    img = io.imread(sys.argv[1] + '/' + str(i) + '.jpg')
    #img = transform.resize(img, (400, 400, 3))
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

y = list(sys.argv[2].split('.'))
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
y -= np.min(y)
y /= np.max(y)
y = (y * 255).astype(np.uint8)
y = y.reshape(600, 600, 3)
#y = transform.resize(y, (600, 600, 3))
io.imsave('reconstruction.png', y)

'''
x_mean = x_mean.reshape(400, 400, 3)
x_mean = transform.resize(x_mean, (600, 600, 3))
io.imsave('mean.jpg', x_mean)

for i in range(4):
    eigen = u[i]
    eigen -= np.min(eigen)
    eigen /= np.max(eigen)
    eigen = (eigen * 255).astype(np.uint8)
    eigen = eigen.reshape(400, 400, 3)
    eigen = transform.resize(eigen, (600, 600, 3))
    io.imsave('eigen' + str(i) + '.png', eigen)
'''
