import streamlit as st
import joblib
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "fake_news_model.pkl")

model = joblib.load(model_path)

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Load model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# NLP setup
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Clean text function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    words = text.split()

    words = [
        lemmatizer.lemmatize(w)
        for w in words
        if w not in stop_words
    ]

    return " ".join(words)

# Streamlit UI
st.set_page_config(page_title="Fake News Detection")

st.title("🧠 Fake News Detection System")

news = st.text_area("Enter News Text")

if st.button("Predict"):

    if news.strip() == "":
        st.warning("Please enter news text.")
    else:
        cleaned = clean_text(news)

        vector = vectorizer.transform([cleaned])

        prediction = model.predict(vector)

        if prediction[0] == 1:
            st.success("🟢 Real News")
        else:
            st.error("🔴 Fake News")
