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

# 35. Compare K-Means and Hierarchical Clustering.
scaled_data = StandardScaler().fit_transform(df)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(scaled_data)

hierarchical = AgglomerativeClustering(n_clusters=3)
hierarchical_labels = hierarchical.fit_predict(scaled_data)

kmeans_score = silhouette_score(scaled_data, kmeans_labels)
hierarchical_score = silhouette_score(
    scaled_data,
    hierarchical_labels
)

comparison = pd.DataFrame({
    "Algorithm": ["K-Means", "Hierarchical"],
    "Silhouette Score": [kmeans_score, hierarchical_score]
})

print(comparison)
print("\nBoth algorithms divide the data into clusters.")
print("Silhouette Score is used here to compare cluster quality.")
