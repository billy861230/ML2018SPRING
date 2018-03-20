import sys
import csv 
import math
import random
import numpy as np

w = np.load('model_best.npy')

test_x = []
n_row = 0
text = open(sys.argv[1] ,"r")
row = csv.reader(text , delimiter= ",")

for r in row:
    if n_row %18 == 9:
        p=[1]
        p.append(float(r[10]))
        for i in range(8):
            if float(r[9-i])>200:
                p.append(float(r[10-i]))
            else:
                p.append(float(r[9-i]))
        x=[]
        for i in range(len(p)):
            x.append(p[len(p)-1-i])
        test_x.append(x)
    n_row = n_row+1
#print(test_x)
text.close()
test_x = np.array(test_x)
print(test_x)
#test_x=np.asmatrix(test_x)
'''
ans=test_x*a
ans=np.array(ans)
ans=list(ans)
print(test_x)
print(ans)
'''
result=[["id","value"]]
i=0

for i in range(len(test_x)):
    a=['id_' + str(i)]
    if np.dot(test_x[i],w)<1:
        a.append(1)
    else:
        a.append(np.dot(test_x[i],w))
    result.append(a)
    i+=1
cout = csv.writer(open(sys.argv[2],'w'))
cout.writerows(result)

