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

# 18. Hierarchical Clustering using Complete linkage.
scaled_data = StandardScaler().fit_transform(df)

model = AgglomerativeClustering(
    n_clusters=3,
    linkage="complete"
)

labels = model.fit_predict(scaled_data)

print("Complete linkage cluster counts:")
print(pd.Series(labels).value_counts().sort_index())
