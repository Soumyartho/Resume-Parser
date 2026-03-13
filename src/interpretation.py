import numpy as np

def print_top_words_per_cluster(model, vectorizer, n_words=10):
    """
    Print top words for each K-Means cluster
    """

    terms = vectorizer.get_feature_names_out()

    for i, center in enumerate(model.cluster_centers_):
        print(f"\n=== Cluster {i} ===")

        # get indices of largest values
        top_indices = center.argsort()[-n_words:][::-1]

        top_words = [terms[ind] for ind in top_indices]

        print("Top words:", ", ".join(top_words))
