import streamlit as st
import pickle
import os

st.set_page_config(page_title="Spam Classifier", page_icon="📩")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model1.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer1.pkl")

# Load model
def load_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        import recreate_pkls

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer

model, vectorizer = load_model()

# UI
st.title("📩 SMS Spam Classifier")
st.write("Check if your message is Spam or Ham")

message = st.text_area("Enter your message:")

if st.button("Predict"):
    if message.strip() == "":
        st.warning("Please enter a message")
    else:
        X = vectorizer.transform([message])
        result = model.predict(X)[0]
        prob = model.predict_proba(X)[0]

        spam_prob = prob[list(model.classes_).index("spam")] * 100
        ham_prob = prob[list(model.classes_).index("ham")] * 100

        if result == "spam":
            st.error(f"🚨 Spam ({spam_prob:.2f}%)")
        else:
            st.success(f"✅ Ham ({ham_prob:.2f}%)")

        # Probability bars
        st.subheader("Prediction Confidence")
        st.progress(int(spam_prob))
        st.write(f"Spam Probability: {spam_prob:.2f}%")