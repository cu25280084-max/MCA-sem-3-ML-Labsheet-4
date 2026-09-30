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

# 28. K-Means clustering on PCA-transformed data.
scaled_data = StandardScaler().fit_transform(df)

pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

model = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = model.fit_predict(pca_data)

plt.scatter(
    pca_data[:, 0],
    pca_data[:, 1],
    c=labels
)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("K-Means on PCA Data")
plt.show()
