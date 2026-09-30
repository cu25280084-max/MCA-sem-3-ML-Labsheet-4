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

# 24. Visualize PCA-transformed data.
scaled_data = StandardScaler().fit_transform(df)

pca = PCA(n_components=2)
result = pca.fit_transform(scaled_data)

plt.scatter(result[:, 0], result[:, 1])
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA 2D Visualization")
plt.show()
