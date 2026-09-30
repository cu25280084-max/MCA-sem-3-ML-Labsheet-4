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

# 6. Implement K-Means clustering.
features = df.copy()
scaled_data = StandardScaler().fit_transform(features)

model = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = model.fit_predict(scaled_data)

print("Cluster labels:")
print(labels)
