import os
import pandas as pd

def download_kaggle_dataset():
    # We use the Kaggle library to download the dataset programmatically.
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("Error: The 'kaggle' library is not installed.")
        print("Please run: pip install kaggle pandas")
        return

    # Attempt to authenticate using the Kaggle API token (~/.kaggle/kaggle.json)
    try:
        api = KaggleApi()
        api.authenticate()
    except Exception as e:
        print("Error during Kaggle Authentication:", e)
        print("\n--- Setup Instructions ---")
        print("1. Log in to Kaggle.com and go to your Account settings.")
        print("2. Scroll down to the 'API' section and click 'Create New API Token'.")
        print("3. This downloads a file named 'kaggle.json'.")
        print(r"4. Create the folder 'C:\Users\Nitro\.kaggle' if it doesn't exist.")
        print(r"5. Move 'kaggle.json' into 'C:\Users\Nitro\.kaggle\kaggle.json'.")
        print("--------------------------")
        return

    # We use a popular Kaggle resume dataset: gauravduttakiit/resume-dataset
    dataset = 'gauravduttakiit/resume-dataset'
    
    # Destination matches your existing data folder
    dest_path = r"c:\Users\Nitro\OneDrive\Desktop\ResumeAI\data"
    os.makedirs(dest_path, exist_ok=True)
    
    print(f"Downloading dataset '{dataset}' into {dest_path}...")
    
    try:
        # Download and automatically unzip the files
        api.dataset_download_files(dataset, path=dest_path, unzip=True)
    except Exception as e:
        print("Failed to download dataset:", e)
        return
    
    # The primary file in this dataset is 'UpdatedResumeDataSet.csv'
    csv_file = os.path.join(dest_path, 'UpdatedResumeDataSet.csv')
    
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        
        print(f"Successfully downloaded {len(df)} real resumes from Kaggle!")
        print(df.head())
        
        # Save as Resume.csv to maintain compatibility with your other scripts
        target_file = os.path.join(dest_path, 'Resume.csv')
        df.to_csv(target_file, index=False)
        print(f"\nSaved the Kaggle data to {target_file} for downstream processing.")
    else:
        print("Download finished, but 'UpdatedResumeDataSet.csv' was not found.")
        print(f"Please check the contents of: {dest_path}")

if __name__ == "__main__":
    download_kaggle_dataset()
