import pandas as pd
from src.preprocess import clean_text
from src.vectorizer import vectorize_text
from src.clustering import run_kmeans
from src.interpretation import print_top_words_per_cluster
from src.scoring import calculate_scores

# Load dataset
df = pd.read_csv("data/Resume.csv")

# Clean text
df["Cleaned_Resume"] = df["Resume"].apply(clean_text)

# TF-IDF Vectorization
X, vectorizer = vectorize_text(df["Cleaned_Resume"])

print("TF-IDF Matrix Shape:", X.shape)

# Run K-Means clustering
labels, model = run_kmeans(X, n_clusters=5)

# Add clusters to dataframe
df["Cluster"] = labels

print("\nCluster counts:")
print(df["Cluster"].value_counts())

# Understand clusters
print_top_words_per_cluster(model, vectorizer)

# Calculate resume scores
df["Score"] = calculate_scores(X, labels, model)

print("\nTop 10 Resumes by Score:")
print(
    df[["Name", "Category", "Cluster", "Score"]]
    .sort_values(by="Score", ascending=False)
    .head(10)
)
