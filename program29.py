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

# 29. Compare clustering before and after PCA.
scaled_data = StandardScaler().fit_transform(df)

kmeans_original = KMeans(n_clusters=3, random_state=42, n_init=10)
original_labels = kmeans_original.fit_predict(scaled_data)

pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

kmeans_pca = KMeans(n_clusters=3, random_state=42, n_init=10)
pca_labels = kmeans_pca.fit_predict(pca_data)

print("Silhouette before PCA:",
      silhouette_score(scaled_data, original_labels))
print("Silhouette after PCA :",
      silhouette_score(pca_data, pca_labels))
