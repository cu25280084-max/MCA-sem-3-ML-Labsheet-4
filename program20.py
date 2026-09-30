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

# 20. Visualize Hierarchical Clustering results.
scaled_data = StandardScaler().fit_transform(df)

model = AgglomerativeClustering(n_clusters=3)
labels = model.fit_predict(scaled_data)

plt.scatter(
    df["petal length (cm)"],
    df["petal width (cm)"],
    c=labels
)
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.title("Hierarchical Clustering")
plt.show()
