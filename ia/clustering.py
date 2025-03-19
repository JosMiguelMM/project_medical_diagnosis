from sklearn.cluster import KMeans
import pandas as pd
import numpy as np

class KMeansCluster:
    def __init__(self, n_clusters=3, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        self.cluster_centers_ = None
        self.labels_ = None

    def fit(self, data):
        if not isinstance(data, pd.DataFrame):
            raise ValueError("La entrada debe ser un DataFrame de pandas.")

        self.kmeans.fit(data)
        self.cluster_centers_ = self.kmeans.cluster_centers_
        self.labels_ = self.kmeans.labels_
        return self

    def predict(self, data):
        if not isinstance(data, pd.DataFrame):
            raise ValueError("La entrada debe ser un DataFrame de pandas.")

        return self.kmeans.predict(data)

    def get_cluster_centers(self):
        return self.cluster_centers_

    def get_labels(self):
        return self.labels_
