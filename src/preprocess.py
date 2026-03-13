import re
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def clean_text(text):
    # Handle NaN or non-string values gracefully
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
        if text.lower() == 'nan':
            return ""

    # lowercase
    text = text.lower()

    # remove numbers and punctuation
    text = re.sub(r'[^a-z\s]', ' ', text)

    # remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # remove stopwords
    words = text.split()
    words = [w for w in words if w not in stop_words]

    return " ".join(words)
