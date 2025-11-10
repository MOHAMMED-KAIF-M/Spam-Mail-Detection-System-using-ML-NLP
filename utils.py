# utils.py
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords', quiet=True)
STOP = set(stopwords.words('english'))

def clean_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r'http\S+|www\S+',' ', s)
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    tokens = [t for t in s.split() if t not in STOP and len(t)>1]
    return " ".join(tokens)
