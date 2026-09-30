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

# 27. Compare dataset before and after dimensionality reduction.
scaled_data = StandardScaler().fit_transform(df)

pca = PCA(n_components=2)
reduced_data = pca.fit_transform(scaled_data)

print("Original shape:", df.shape)
print("Reduced shape :", reduced_data.shape)
