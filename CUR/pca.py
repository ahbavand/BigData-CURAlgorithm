import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors



csv = np.genfromtxt ('/Users/amirhossein/Desktop/term7/Big_Data/projects/HW1/pat.csv', delimiter=",")
csv=np.transpose(csv)

a=csv

pca = PCA(n_components=20, svd_solver='full')

y=pca.fit_transform(a)

neigh = NearestNeighbors(6)
neigh.fit(a)

w=neigh.kneighbors(a[50000:51000])[1]

print(w)


neigh = NearestNeighbors(6)
neigh.fit(y)

v=neigh.kneighbors(y[50000:51000])[1]

z = 0
for i in range(0, 1000):
    for j in range(1, 6):
        if (v[i][j] in w[i]):
            z = z + 1

print(z)




