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

# 19. Compare different linkage methods.
scaled_data = StandardScaler().fit_transform(df)

for method in ["ward", "complete", "average", "single"]:
    model = AgglomerativeClustering(
        n_clusters=3,
        linkage=method
    )
    labels = model.fit_predict(scaled_data)

    score = silhouette_score(scaled_data, labels)
    print(f"{method.title()} linkage: {score:.4f}")
