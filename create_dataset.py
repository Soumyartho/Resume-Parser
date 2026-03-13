import os
from docx import Document
import pandas as pd

def parse_resumes_to_csv(resume_folder, output_csv):
    """
    Reads all .docx resumes in the given folder, extracts the text,
    attempts to guess the category from the filename, and saves
    everything to a CSV formatted for the ResumeAI pipeline.
    """
    data = []

    if not os.path.exists(resume_folder):
         print(f"Error: The folder '{resume_folder}' does not exist.")
         return

    print(f"Scanning '{resume_folder}' for Word documents...")

    for filename in os.listdir(resume_folder):
        if filename.endswith(".docx"):
            filepath = os.path.join(resume_folder, filename)
            
            try:
                # Read the Word Document
                doc = Document(filepath)
                text = " ".join([paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip() != ""])
                
                # Clean the filename to extract a readable candidate name
                name_clean = filename.rsplit(".", 1)[0]  # Remove extension
                name_clean = name_clean.replace("_", " ").replace("-", " ")
                
                # Remove noisy keywords to isolate the name
                noisy_words = {"resume", "profile", "sr", "bsa", "sm", "fullstack", "java", "pmp1", "developer", "cv", "updated", "j2ee"}
                words = name_clean.split()
                clean_words = []
                for w in words:
                    if w.lower() not in noisy_words and not any(char.isdigit() for char in w):
                        clean_words.append(w)
                        if len(clean_words) == 2:  # typically First Last name
                            break
                            
                candidate_name = " ".join(clean_words).title() if clean_words else "Unknown Candidate"

                # Guess a category based on keywords in the filename
                name_lower = filename.lower()
                if "java" in name_lower or "j2ee" in name_lower:
                    category = "Java Developer"
                elif "ba" in name_lower or "business analyst" in name_lower:
                    category = "Business Analyst"
                elif "pm" in name_lower or "project" in name_lower:
                    category = "Project Manager"
                elif "data" in name_lower or "science" in name_lower or "machine learning" in name_lower:
                    category = "Data Science"
                elif "hr" in name_lower or "human" in name_lower:
                    category = "HR"
                elif "engineer" in name_lower:
                    category = "Engineer"
                else:
                    category = "Other" # Default if we can't guess from the name
                
                data.append({"Name": candidate_name, "Category": category, "Resume": text})
            except Exception as e:
                print(f"Failed to read {filename}: {e}")

    if not data:
        print("No resumes were successfully parsed. Did you check the folder path and ensure they are .docx files?")
        return

    # Convert to DataFrame and Save
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    
    print(f"\nSuccessfully parsed {len(df)} resumes!")
    print(f"Saved to: {output_csv}")
    print("\nSample of generated data:")
    print(df.head())

if __name__ == "__main__":
    # --- CONFIGURE THESE PATHS ---
    # Update this path if the user moved the folder somewhere else
    RESUME_FOLDER_PATH = r"C:\Users\Nitro\Downloads\archive (1)\Resumes"
 
    
    # Save directly to the data folder so main.py can use it
    OUTPUT_CSV_PATH = r"c:\Users\Nitro\OneDrive\Desktop\ResumeAI\data\Resume.csv"
    
    parse_resumes_to_csv(RESUME_FOLDER_PATH, OUTPUT_CSV_PATH)
