# train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from utils import clean_text

DATA_PATH = "data/spam.csv"
MODEL_OUT = "models/spam_model.joblib"

def load_data(path):
    df = pd.read_csv(path)
    # normalize column names
    if 'label' not in df.columns:
        raise ValueError("CSV must have 'label' column")
    df = df.dropna(subset=['text'])
    df['text'] = df['text'].astype(str).map(clean_text)
    df['label_num'] = df['label'].map(lambda x: 1 if str(x).lower().startswith('s') else 0)
    return df['text'], df['label_num']

def main():
    X, y = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=20000, ngram_range=(1,2))),
        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
    ])

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    print("=== Classification report ===")
    print(classification_report(y_test, preds, target_names=['ham','spam']))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, preds))

    joblib.dump(pipe, MODEL_OUT)
    print(f"Model saved to {MODEL_OUT}")

if __name__ == "__main__":
    main()
