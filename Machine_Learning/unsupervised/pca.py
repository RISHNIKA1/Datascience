import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

data = load_iris()
X = data.data
y= data.target

#Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#PCA
pca =PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Explained Variance :",pca.explained_variance_ratio_)

plt.scatter(X_pca[:,0],X_pca[:,1],c=y)
plt.title("PCA -2D Visualization")
plt.show()