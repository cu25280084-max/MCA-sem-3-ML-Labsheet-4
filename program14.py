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

# 14. Compare clustering before and after scaling.
unscaled_model = KMeans(n_clusters=3, random_state=42, n_init=10)
unscaled_labels = unscaled_model.fit_predict(df)

scaled_data = StandardScaler().fit_transform(df)
scaled_model = KMeans(n_clusters=3, random_state=42, n_init=10)
scaled_labels = scaled_model.fit_predict(scaled_data)

print("Unscaled silhouette:",
      silhouette_score(df, unscaled_labels))
print("Scaled silhouette  :",
      silhouette_score(scaled_data, scaled_labels))
