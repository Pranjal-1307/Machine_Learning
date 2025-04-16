#USING ARRAY
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

x= np.array([[185,72],
	[170,56],
	[168,60],
	[179,68],
	])

kmeans=KMeans(n_clusters=2,random_state=42,n_init=10)
kmeans.fit(x)
labels=kmeans.labels_
centroids=kmeans.cluster_centers_

print("Cluster Labels:",labels)

plt.figure(figsize=(8,6))
plt.scatter(x[:, 0], x[:, 1], c='green',  marker='o', label="Data Points") 
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label="Centroids")
plt.xlabel("Height") 
plt.ylabel("Weight") 
plt.title("K-Means Clustering (K=2)") 
plt.legend() 
plt.grid(True) 
plt.show()



#USING CSV 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data=pd.read('file.csv')
x=data[['height','weight']].values

kmeans=KMeans(n_clusters=2,random_state=4,n_init=10)
kmeans.fit(x)
labels=kmeans.labels_
centroids=kmeans.cluster_centers_

print("Cluster:",labels)

plt.figure(figsize=(8,6))
plt.scatter(x[:,0],x[:,1],c='green',marker='o',label='')
plt.scatter(centroids[:,0],centroids[:,1],c='green',marker='x',label='')
plt.xlabel("Height") 
plt.ylabel("Weight") 
plt.title("K-Means Clustering (K=2)") 
plt.legend() 
plt.grid(True) 
plt.show()
