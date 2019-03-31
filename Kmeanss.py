# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 00:12:32 2018

@author: nakul
"""

#Kmeans Clustering
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('Mall_Customers.csv')

X = dataset.iloc[:,[3,4]].values

#Use elbow method to find optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', n_init = 10, max_iter = 300, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.title('The Elbow Method')
plt.xlabel('No of Clusters')
plt.ylabel('WCSS')
plt.legend()
plt.show()

#We found 5 CLusters

#Applying Kmeans with 5 clusters
kmeans = KMeans(n_clusters = 5, init = 'k-means++', n_init = 10, max_iter = 300, random_state = 0)
y_kmeans = kmeans.fit_predict(X)

#Visualizing the Result
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0,1], s= 100, c = 'red', label = 'Cluster1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1,1], s= 100, c = 'green', label = 'Cluster2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2,1], s= 100, c = 'blue', label = 'Cluster3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3,1], s= 100, c = 'orange', label = 'Cluster4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4,1], s= 100, c = 'cyan', label = 'Cluster5')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s = 600, c = 'yellow', label = 'centroid')
plt.title('The Elbow Method')
plt.xlabel('No of Clusters')
plt.ylabel('WCSS')
plt.legend()
plt.show()

#Hierarchial CLustering
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('Mall_Customers.csv')

X = dataset.iloc[:,[3,4]].values

#Use Dendogram to find Optimal number of CLusters
import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendogram')
plt.xlabel('No of Clusters')
plt.ylabel('Euclidean Distances')
plt.show()

#Fitting CLuster onto Model
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)

plt.scatter(X[y_hc == 0, 0], X[y_hc == 0,1], s= 100, c = 'red', label = 'Cluster1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1,1], s= 100, c = 'green', label = 'Cluster2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2,1], s= 100, c = 'blue', label = 'Cluster3')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3,1], s= 100, c = 'orange', label = 'Cluster4')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4,1], s= 100, c = 'cyan', label = 'Cluster5')
plt.title('The Elbow Method')
plt.xlabel('No of Clusters')
plt.ylabel('WCSS')
plt.legend()
plt.show()
