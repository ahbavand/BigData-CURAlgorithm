import numpy as np
import math
import random
import math
from sklearn.neighbors import NearestNeighbors



csv = np.genfromtxt ('/Users/amirhossein/Desktop/term7/Big_Data/projects/HW1/pat.csv', delimiter=",")
csv=np.transpose(csv)


b=csv
a=b[2000:22000]
k=20     #selected_rank
row_size=np.size(a,0)
column_size=np.size(a,1)
print(column_size)


U, s, V = np.linalg.svd(a, full_matrices=True)


p=np.zeros(column_size)

for i in range(0,column_size):
    for j in range(0,k):
        p[i]=p[i]+math.pow(V[i][j],2)/k



p1=np.zeros(column_size)

p1[0]=p[0]
for i in range(1,column_size):
    p1[i]=p1[i-1]+p[i]




C=np.zeros((row_size,k))

for q in range(0, k):

    randomnumber = (random.random())

    boolean = True

    i = 0

    columnnumber = 0
    while (boolean):
        if (randomnumber < p1[i]):
            columnnumber = i
            break

        i = i + 1

    for d in range(0, row_size):
        C[d][q] = a[d][columnnumber]
# end of constructing C






#constructing   R

atrans=a.T
U, s, V = np.linalg.svd(atrans, full_matrices=True)



p=np.zeros(row_size)

for i in range(0,row_size):
    for j in range(0,k):
        p[i]=p[i]+math.pow(V[i][j],2)/k



p1=np.zeros(row_size)

p1[0]=p[0]
for i in range(1,row_size):
    p1[i]=p1[i-1]+p[i]



R=np.zeros((column_size,k))


for q in range(0,k):

    randomnumber = (random.random())


    boolean=True

    i=0

    columnnumber=0
    while(boolean):
        if(randomnumber<p1[i]):
            columnnumber=i
            break

        i=i+1

    for d in range(0,column_size):
        R[d][q]=atrans[d][columnnumber]



R=R.T
# end of constructing R




C_moore_penrose=np.linalg.pinv(C)

R_moore_penrose=np.linalg.pinv(R)


utemp=np.matmul(C_moore_penrose,a)
U=np.matmul(utemp,R_moore_penrose)





i=np.matmul(np.matmul(C,U),R)

print(i)

print(i.shape)




a_dim_reduction=np.matmul(a,R.T)
print(a_dim_reduction.shape)




neigh = NearestNeighbors(6)
neigh.fit(a)




w=neigh.kneighbors(a[2000:3000])[1]


neigh = NearestNeighbors(6)
neigh.fit(a_dim_reduction)


v=neigh.kneighbors(a_dim_reduction[2000:3000])[1]

z = 0
for i in range(0, 1000):
    for j in range(1, 6):
        if (v[i][j] in w[i]):
            z = z + 1

print(z)















