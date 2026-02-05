import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=300, centers=1, cluster_std=0.7, random_state=42)

# add outliers
outliers = np.random.uniform(low=-6, high=6, size=(20, 2))
X = np.vstack([X, outliers])

model = IsolationForest(contamination=0.06, random_state=42)
pred = model.fit_predict(X)

# -1 = outlier, 1 = normal
plt.scatter(X[:, 0], X[:, 1], c=pred)
plt.title("Isolation Forest Outlier Detection")
plt.show()
