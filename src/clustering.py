from sklearn.neighbors import NearestNeighbors

def run_knn_search(X, query_vector, n_neighbors=10):
    """
    Finds the strictly K nearest neighbors to a query profile vector using Cosine distance.
    """
    # Ensure n_neighbors isn't larger than the dataset
    n_neighbors = min(n_neighbors, X.shape[0])
    
    # We use cosine metric for text similarity
    model = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
    model.fit(X)
    
    # Return both the distances and the indices of the matches
    distances, indices = model.kneighbors(query_vector)
    
    return distances[0], indices[0], model
