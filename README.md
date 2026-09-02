# 📩 SMS Spam Detection Classifier

An end-to-end Machine Learning web application designed to classify SMS messages as either **Spam** or **Ham** (Not Spam) using Natural Language Processing (NLP) techniques.

---

## 🚀 Live Demo

Check out the live deployed application here:  
👉 **[Click Here to View Live App](https://spam-msg-predictor.streamlit.app/)**


---

## 📌 Features

- **Real-Time Classification:** Instantly detects whether an entered text message is spam or legitimate.
- **NLP Preprocessing Pipeline:**
  - Lowercasing and tokenization
  - Special character and punctuation removal
  - Stopword filtering
  - Stemming using NLTK
- **Interactive UI:** Built with an intuitive, clean interface using Streamlit.
- **Pre-trained Model:** Fast inference powered by TF-IDF Vectorization and Naive Bayes / Classification algorithms.

---

## 🛠️ Tech Stack

- **Language:** Python
- **Libraries:** Scikit-learn, NLTK, Pandas, NumPy
- **Frontend / Framework:** Streamlit
- **Model Serialization:** Pickle

---

## 📁 Repository Structure

```text
sms-spam-classifier/
├── app.py                  # Main Streamlit application
├── model.pkl               # Trained classification model
├── vectorizer.pkl          # Fitted TF-IDF Vectorizer
├── spam.csv                # SMS Spam Collection dataset (optional)
├── sms_spam_classifier.ipynb # Jupyter notebook with EDA & model training
├── requirements.txt        # Required Python packages
└── README.md               # Project documentation
