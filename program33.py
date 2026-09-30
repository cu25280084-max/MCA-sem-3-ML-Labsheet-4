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

# 33. Visualize cluster distributions using a pair plot.
scaled_data = StandardScaler().fit_transform(df)

model = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = model.fit_predict(scaled_data)

result = df.copy()
result["Cluster"] = labels.astype(str)

sns.pairplot(result, hue="Cluster")
plt.show()
