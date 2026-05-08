import streamlit as st
import joblib
import re
import nltk
import os

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -------------------------
# NLTK setup
# -------------------------
nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# -------------------------
# Safe file loading (IMPORTANT)
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "fake_news_model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "tfidf_vectorizer.pkl")

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# -------------------------
# Clean text function
# -------------------------
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

# -------------------------
# Streamlit UI
# -------------------------
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
