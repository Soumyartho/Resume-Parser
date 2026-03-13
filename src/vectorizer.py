from sklearn.feature_extraction.text import TfidfVectorizer

def vectorize_text(text_column):
    """
    Converts cleaned resume text into TF-IDF vectors
    """

    vectorizer = TfidfVectorizer(
        max_features=1000,   # limit features (safe for beginners)
        stop_words='english'
    )

    X = vectorizer.fit_transform(text_column)

    return X, vectorizer
