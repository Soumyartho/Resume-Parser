import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def calculate_scores(X, labels, model):
    """
    Calculates resume score (0-10) based on similarity
    to cluster centroid, normalized within each cluster
    so top candidates from every category can score highly.
    """

    raw_scores = []

    for i in range(X.shape[0]):
        # resume vector
        resume_vector = X[i]

        # corresponding cluster center
        cluster_id = labels[i]
        centroid = model.cluster_centers_[cluster_id]

        # cosine similarity (0 → 1)
        sim = cosine_similarity(
            resume_vector,
            centroid.reshape(1, -1)
        )[0][0]

        raw_scores.append(sim)

    # Convert to numpy array for easier indexing
    normalized = np.zeros(len(raw_scores))
    labels_arr = np.array(labels)
    
    # Normalize to 0-10 scale WITHIN each cluster
    for cluster_id in np.unique(labels_arr):
        indices = np.where(labels_arr == cluster_id)[0]
        cluster_scores = [raw_scores[i] for i in indices]
        
        min_s = min(cluster_scores)
        max_s = max(cluster_scores)
        
        # Avoid division by zero
        if max_s > min_s:
            for i in indices:
                normalized[i] = round((raw_scores[i] - min_s) / (max_s - min_s) * 10, 2)
        else:
            for i in indices:
                normalized[i] = 10.0

    return list(normalized)
