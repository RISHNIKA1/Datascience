import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

X,y=make_blobs(n_samples=300,centers=4,random_state=42)

kmeans = KMeans(n_clusters=4,random_state=42)
kmeans.fit(X)

labels = kmeans.labels_
centroids =kmeans.cluster_centers_

plt.scatter(X[:,0],X[:,1],c=labels)
plt.scatter(centroids[:,0],centroids[:,1],marker="X",s=200)
plt.title("K-Means Clustering")
plt.show()