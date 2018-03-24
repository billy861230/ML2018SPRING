import sys
import csv 
import math
import random
import numpy as np

data = []    

#一個維度儲存一種污染物的資訊
for i in range(18):
	data.append([])

n_row = 0
text = open('data/train.csv', 'r', encoding='big5') 
row = csv.reader(text , delimiter=",")
for r in row:
    # 第0列為header沒有資訊
    if n_row != 0:
        # 每一列只有第3-27格有值(1天內24小時的數值)
        for i in range(3,27):
            if r[i] != "NR":
                data[(n_row-1)%18].append(float(r[i]))
            else:
                data[(n_row-1)%18].append(float(0))	
    n_row = n_row+1
text.close()

x = []
y = []

for i in range(12):
    # 一個月取連續10小時的data可以有471筆
    for j in range(471):
        b=True
        z=[]
        # 總共有18種污染物
        #for t in range(18):
            # 連續9小時
        for s in range(9):
            if data[9][480*i+j+s]>200:
                b=False
                break
            z.append(data[9][480*i+j+s])
        if b:
            z.append(1)
            x.append(z)
            y.append(data[9][480*i+j+9])
x = np.array(x)
z=[]
for i in range(len(y)):
    z.append([])
    z[i].append(y[i])
x=np.asmatrix(x)
y=np.asmatrix(z)
a=(((x.T*x).I)*x.T)*y
a=np.array(a)
w=[]
for i in range(10):
    w.append(a[i][0])
np.save('model_best.npy',w)



# x = np.concatenate((x,x**2), axis=1)
# 增加平方項
