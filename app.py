import streamlit as st
import pandas as pd
import altair as alt
import os

# Set page config
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide"
)

# Custom CSS for a cleaner look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stMetric {
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.5);
    }
    </style>
""", unsafe_allow_html=True)

# Application Header
st.title("📄 AI Resume Analyzer")
st.markdown("Automatically screen, cluster, and rank resumes using Natural Language Processing and Machine Learning.")
st.markdown("---")

# Error loading data check
@st.cache_data
def load_data():
    if not os.path.exists("data/Resume.csv"):
        st.error("Dataset not found! Please run `create_dataset.py` first.")
        return None
    return pd.read_csv("data/Resume.csv")

# Run pipeline if requested
def run_pipeline():
    from src.preprocess import clean_text
    from src.vectorizer import vectorize_text
    from src.clustering import run_kmeans
    from src.scoring import calculate_scores
    
    with st.spinner("Analyzing resumes using NLP & K-Means..."):
        df = load_data()
        if df is None: return None
        
        # We handle NaN values here just in case the pipeline crashed previously
        df["Cleaned_Resume"] = df["Resume"].fillna("").astype(str).apply(clean_text)
        X, vectorizer = vectorize_text(df["Cleaned_Resume"])
        
        # User defined clusters
        num_clusters = st.sidebar.slider("Number of Skill Clusters to Find:", min_value=3, max_value=10, value=5)
        
        labels, model = run_kmeans(X, n_clusters=num_clusters)
        df["Cluster"] = labels
        df["Score"] = calculate_scores(X, labels, model)
        
        return df, model, vectorizer

# Sidebar Controls
st.sidebar.header("⚙️ Configuration")
if st.sidebar.button("🚀 Run AI Pipeline", use_container_width=True, type="primary"):
    st.session_state['results'] = run_pipeline()
    st.success("Analysis Complete!")

st.sidebar.markdown("---")
st.sidebar.markdown("### Display Settings")
top_n = st.sidebar.number_input("How many top candidates to show?", min_value=5, max_value=50, value=10)

# Main Dashboard Content
if 'results' in st.session_state and st.session_state['results'] is not None:
    df, model, vectorizer = st.session_state['results']
    
    # --- Top Metrics Row ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Resumes Scanned", len(df))
    col2.metric("Skill Clusters Found", df["Cluster"].nunique())
    col3.metric("Highest Category Count", df["Category"].value_counts().index[0])
    
    st.markdown("---")
    
    # --- Analysis Parameters ---
    st.subheader("⚙️ Analysis Parameters")
    st.markdown("""
    The resumes below were analyzed, grouped, and ranked using the following settings:
    * **Text Pre-processing:** Removed punctuation, numbers, and common stop words using the **NLTK** library.
    * **Feature Extraction:** Extracted relevant skills by converting text to numerical vectors using **TF-IDF**.
    * **Grouping:** Automatically clustered candidates into skill domains using **K-Means Clustering**.
    * **Scoring:** Ranked candidates based on **Cosine Similarity** to the ideal profile in their cluster (normalized from 0 to 10 scale).
    """)

    # --- Top Candidates Table ---
    st.markdown("---")
    st.subheader("🏆 Top Ranked Candidates")
    
    # Create the display dataframe
    display_df = df[["Name", "Category", "Cluster", "Score"]].copy()
    display_df = display_df.sort_values(by="Score", ascending=False).head(top_n)
    
    # Format the score for better display
    display_df["Score"] = display_df["Score"].apply(lambda x: f"{x:.2f} / 10.00")
    display_df.index = range(1, len(display_df) + 1) # Clean 1-index for presentation
    
    st.dataframe(
        display_df,
        use_container_width=True,
        column_config={
            "Name": st.column_config.TextColumn("Candidate Name", width="medium"),
            "Category": st.column_config.TextColumn("Job Category"),
            "Cluster": st.column_config.NumberColumn("AI Cluster Group"),
            "Score": st.column_config.TextColumn("Relevance Score")
        }
    )
    
    # --- Interactive Resume Viewer ---
    st.markdown("---")
    st.subheader("🔍 Review Individual Resume")
    
    # Dropdown to select a top candidate to read
    candidate_options = display_df["Name"].tolist()
    selected_candidate = st.selectbox("Select a top candidate to view their raw resume text:", candidate_options)
    
    if selected_candidate:
        resume_text = df[df["Name"] == selected_candidate].iloc[0]["Resume"]
        
        # Display the text in a clean scrollable box
        with st.expander(f"Raw Resume Text for: {selected_candidate}", expanded=True):
            st.text_area("", resume_text, height=300, disabled=True)
        
else:
    # Empty State
    st.info("👋 Welcome! Click the **🚀 Run AI Pipeline** button in the sidebar to process the dataset and view the dashboard.")
