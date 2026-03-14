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

from src.preprocess import clean_text
from src.vectorizer import vectorize_text
from src.clustering import run_knn_search
from src.scoring import calculate_match_scores

# Error loading data check
@st.cache_data
def load_data():
    if not os.path.exists("data/Resume.csv"):
        st.error("Dataset not found! Please run `create_dataset.py` first.")
        return None
    return pd.read_csv("data/Resume.csv")

# Run pipeline if requested
def run_pipeline(selected_skills):
    
    with st.spinner("Analyzing resumes using NLP & KNN..."):
        df = load_data()
        if df is None: return None
        if not selected_skills:
            st.warning("Please select at least one skill to match against.")
            return None
        
        # We handle NaN values here just in case the pipeline crashed previously
        df["Cleaned_Resume"] = df["Resume"].fillna("").astype(str).apply(clean_text)
        X, vectorizer = vectorize_text(df["Cleaned_Resume"])
        
        # Vectorize the required skills string into an Ideal Profile
        query_text = " ".join(selected_skills).lower()
        query_vector = vectorizer.transform([query_text])
        
        distances, indices, model = run_knn_search(X, query_vector, n_neighbors=len(df))
        scores = calculate_match_scores(distances)
        
        # Reorder dataframe based on KNN similarity
        matched_df = df.iloc[indices].copy()
        matched_df["Score"] = scores
        matched_df.reset_index(drop=True, inplace=True)
        
        return matched_df, vectorizer, query_text

AVAILABLE_SKILLS = [
    "Python", "Java", "C++", "JavaScript", "React", "Angular", "Node.js", 
    "AWS", "GCP", "Azure", "Docker", "Kubernetes", "SQL", "NoSQL",
    "Machine Learning", "Data Analysis", "Data Science", "Natural Language Processing", 
    "Project Management", "Agile", "Scrum", "Business Analysis", "Communication",
    "Leadership", "Finance", "Marketing", "Sales", "Customer Service"
]

# Sidebar Controls
st.sidebar.header("⚙️ Configuration")
selected_skills = st.sidebar.multiselect(
    "Select Required Skills (Parameters):",
    options=AVAILABLE_SKILLS,
    default=["Python", "Data Analysis", "Machine Learning"]
)

if st.sidebar.button("🚀 Run AI Pipeline", use_container_width=True, type="primary"):
    st.session_state['results'] = run_pipeline(selected_skills)
    st.success("Analysis Complete!")

st.sidebar.markdown("---")
st.sidebar.markdown("### Display Settings")
top_n = st.sidebar.number_input("How many top candidates to show?", min_value=5, max_value=50, value=10)

# Main Dashboard Content
if 'results' in st.session_state and st.session_state['results'] is not None:
    df, vectorizer, query_text = st.session_state['results']
    
    # --- Top Metrics Row ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Resumes Scanned", len(df))
    col2.metric("Parameters Searched", len(selected_skills))
    col3.metric("Highest Match Score", f"{df['Score'].max():.2f}")
    
    st.markdown("---")
    
    # --- Analysis Parameters ---
    st.subheader("⚙️ Analysis Parameters")
    st.markdown(f"""
    The resumes below were analyzed and ranked using the following exact parameters:
    * **Target Profile:** Searched for candidates matching: `{query_text}`
    * **Text Pre-processing:** Removed punctuation, numbers, and common stop words using **NLTK**.
    * **Feature Extraction:** Extracted relevant skills by converting text to numerical vectors using **TF-IDF**.
    * **Searching:** Fetched closest candidates using explicitly requested attributes mapped with the **K-Nearest Neighbors (KNN)** algorithm.
    * **Scoring:** Ranked candidates based on **Cosine Similarity** to the exact profile you defined above.
    """)

    # --- Top Candidates Table ---
    st.markdown("---")
    st.subheader("🏆 Top Ranked Candidates")
    
    # Create the display dataframe
    display_df = df[["Name", "Score"]].copy().head(top_n)
    
    # Format the score for better display
    display_df["Score"] = display_df["Score"].apply(lambda x: f"{x:.2f} / 10.00")
    display_df.index = range(1, len(display_df) + 1) # Clean 1-index for presentation
    
    st.dataframe(
        display_df,
        use_container_width=True,
        column_config={
            "Name": st.column_config.TextColumn("Candidate Name", width="medium"),
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
    st.info("👋 Welcome! Select your desired parameters in the sidebar and click **🚀 Run AI Pipeline** to begin candidate matching.")
