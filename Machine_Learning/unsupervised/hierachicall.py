import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from scipy.cluster.hierarchy import dendrogram,linkage
from sklearn.cluster import AgglomerativeClustering #Divisise

X,y = make_blobs(n_samples=300,centers=4,random_state=42)

#dendrogram
linked = linkage(X,method ="ward")
plt.figure(figsize=(10,5))
dendrogram(linked)
plt.title("Dendrogram")
plt.show()

#clustering
hc = AgglomerativeClustering(n_clusters=4)
labels =hc.fit_predict(X)

plt.scatter(X[:,0],X[:,1],c=labels)
plt.title("Hierachical Clustering")
plt.show()