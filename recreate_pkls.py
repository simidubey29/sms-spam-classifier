# recreate_pkls.py
import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "spam.csv")

# Load dataset
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError("❌ spam.csv not found. Please add real dataset.")

print("[INFO] Loading dataset...")
df = pd.read_csv(CSV_PATH, encoding='latin-1')

# Fix column names (for real SMS dataset)
if 'v1' in df.columns and 'v2' in df.columns:
    df = df[['v1', 'v2']]
    df.columns = ['label', 'text']
elif 'label' in df.columns and 'text' in df.columns:
    df = df[['label', 'text']]
else:
    raise RuntimeError(f"Unexpected columns: {df.columns}")

# Clean data
df['text'] = df['text'].astype(str)
df = df.dropna()

texts = df['text']
labels = df['label']

print("[INFO] Training model...")

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X = vectorizer.fit_transform(texts)

# Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X, labels)

# Save files
vpath = os.path.join(BASE_DIR, "vectorizer1.pkl")
mpath = os.path.join(BASE_DIR, "model1.pkl")

with open(vpath, "wb") as f:
    pickle.dump(vectorizer, f)

with open(mpath, "wb") as f:
    pickle.dump(model, f)

print("[SUCCESS] Model trained with real dataset!")
print("[SUCCESS] vectorizer1.pkl & model1.pkl saved")