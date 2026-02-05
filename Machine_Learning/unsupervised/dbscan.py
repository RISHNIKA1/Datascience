import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons

X,y = make_moons(n_samples=200,noise=0.08,random_state=42)

dbscan= DBSCAN(eps = 0.2,min_samples=5)
labels = dbscan.fit_predict(X)

plt.scatter(X[:,0],X[:,1],c=labels)
plt.title("DBSCAN Clustering")
plt.show()