import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

# Iris dataset is used for the clustering experiments.
iris = load_iris(as_frame=True)
df = iris.frame
df = df.drop(columns=["target"])

# 26. Determine cumulative explained variance.
scaled_data = StandardScaler().fit_transform(df)

pca = PCA()
pca.fit(scaled_data)

cumulative = np.cumsum(pca.explained_variance_ratio_)

for i, value in enumerate(cumulative, start=1):
    print(f"Components {i}: {value:.4f}")

plt.plot(
    range(1, len(cumulative) + 1),
    cumulative,
    marker="o"
)
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Cumulative Explained Variance")
plt.show()
