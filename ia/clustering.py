import pandas as pd
from sklearn.cluster import KMeans

def realizar_clustering(datos, n_clusters=3):
    modelo = KMeans(n_clusters=n_clusters, random_state=42)
    etiquetas = modelo.fit_predict(datos)
    return etiquetas
    print("Clustering realizado", pd)
