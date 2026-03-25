# 📄 AI Resume Analyzer

An intelligent, machine-learning-powered application designed to automate the screening, clustering, and ranking of resumes. Built with **Python** and **Streamlit**, this project leverages Natural Language Processing (NLP) to extract meaningful features from raw resume text and uses K-Nearest Neighbors (KNN) to score candidates against a desired job profile.

## ✨ Features

- **Automated Text Preprocessing:** Cleans chaotic resume data by standardizing case, removing punctuation, numbers, and NLTK stop-words.
- **Mathematical Feature Extraction:** Converts qualitative text into quantitative, high-dimensional arrays using **TF-IDF (Term Frequency-Inverse Document Frequency)** vectorization.
- **Smart Candidate Matching:** Treats selected job skills as a "query vector" and uses **Cosine Similarity** via **K-Nearest Neighbors (KNN)** to mathematically rank the relevance of applicants.
- **Interactive Dashboard:** A clean, easy-to-use **Streamlit** user interface that allows recruiters to dynamically select skills, view match percentages, interact with top candidates, and read their raw source resumes.

## 📂 Project Structure

- `app.py`: The main entry point for the Streamlit web application dashboard.
- `main.py`: The backend ML pipeline executing clustering (K-Means) and manual scoring logic.
- `create_dataset.py` / `download_data.py`: Scripts to fetch and generate the underlying `Resume.csv` dataset.
- `src/`: Contains core modular scripts including:
  - `preprocess.py` (Text cleaning using NLP)
  - `vectorizer.py` (TF-IDF logic)
  - `clustering.py` (KNN and K-Means algorithms)
  - `scoring.py` (Cosine similarity scoring logic)
- `requirements.txt`: Project dependencies.

## 🚀 How to Run Locally

Follow these exact steps in your terminal (Command Prompt or PowerShell) to execute the project:

**1. Navigate to the project directory**
```bash
cd c:\Users\Nitro\OneDrive\Desktop\ResumeAI
```

**2. Activate the Virtual Environment**
Ensure you are using the local virtual environment where dependencies are isolated.
```bash
.\venv\Scripts\activate
```

**3. Install Dependencies**
Install all required Python packages (such as `pandas`, `scikit-learn`, `streamlit`, `nltk`).
```bash
pip install -r requirements.txt
```

**4. Generate the Dataset**
If the `data/Resume.csv` file is missing, you must generate or fetch it first.
```bash
python create_dataset.py
```

**5. Launch the Application**
Start the Streamlit UI. This will open a local web server (typically running at `http://localhost:8501`).
```bash
streamlit run app.py
```

## 🧠 Usage Insights

To use the dashboard efficiently:
1. Open the **Configuration Sidebar**.
2. Select your desired candidate parameters (e.g., "Python", "Machine Learning", "Data Analysis").
3. Click the **Run AI Pipeline** button.
4. Review the overall metrics, the generated parameter rundown, and interact with the **Top Ranked Candidates** table to read the highest-scoring resumes immediately.
