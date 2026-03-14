import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def calculate_match_scores(distances):
    """
    Calculates resume score (0-10) based on cosine similarity
    distance returned by KNN.
    Since NearestNeighbors returns cosine distance (1 - similarity),
    we convert it back to similarity, and then scale relative to the best match.
    """
    
    similarities = []
    
    for dist in distances:
        # Cosine distance ranges from 0 (identical) to 2 (opposite).
        # Similarity = max(0, 1 - distance)
        similarity = max(0, 1 - dist)
        similarities.append(similarity)
        
    # Find the best match in the current search query
    max_sim = max(similarities) if similarities else 0
    
    scores = []
    if max_sim > 0.0:
        # Scale everyone directly relative to the best candidate being a 10.0
        for sim in similarities:
            scaled_score = round((sim / max_sim) * 10.0, 2)
            scores.append(scaled_score)
    else:
        # If no one matched at all, just return zeros
        scores = [0.0] * len(similarities)
        
    return scores
