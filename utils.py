import pickle
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"
DATA_PATH = "spam.csv"  # your dataset CSV

def train_and_save_model():
    # Load dataset
    df = pd.read_csv(DATA_PATH, encoding='latin-1')[['v1','v2']]
    df.columns = ['label', 'message']
    df['label'] = df['label'].map({'ham':0, 'spam':1})

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        df['message'], df['label'], test_size=0.2, random_state=42
    )

    # Vectorizer
    vectorizer = CountVectorizer()
    X_train_vect = vectorizer.fit_transform(X_train)

    # Train model
    model = MultinomialNB()
    model.fit(X_train_vect, y_train)

    # Save
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

    return model, vectorizer

def load_model():
    # If files exist, load them; else train and save
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(VECTORIZER_PATH, "rb") as f:
            vectorizer = pickle.load(f)
    else:
        model, vectorizer = train_and_save_model()
    return model, vectorizer