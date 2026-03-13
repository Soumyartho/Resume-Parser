from sklearn.cluster import KMeans

def run_kmeans(X, n_clusters=5):
    """
    Apply K-Means clustering on TF-IDF vectors
    """

    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X)

    return labels, model
