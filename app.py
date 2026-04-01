import streamlit as st
import pickle
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model1.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer1.pkl")

# Load model and vectorizer
def load_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        import recreate_pkls  # will create vectorizer1.pkl and model1.pkl if missing
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_model()

# Streamlit UI
st.title("SMS Spam Classifier 🚀")

msg = st.text_input("Enter your message here:")

if st.button("Predict"):
    if msg.strip() == "":
        st.warning("Please type a message to classify.")
    else:
        X = vectorizer.transform([msg])
        pred = model.predict(X)[0]
        st.success(f"Prediction: **{pred.upper()}**")