**STEP 1**: Install Requirements
"""

pip install pandas numpy scikit-learn nltk matplotlib seaborn

"""**STEP 2**: Import Libraries"""

import pandas as pd
import numpy as np
import re
import nltk

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

"""**STEP 3**: Load Dataset"""

fake = pd.read_csv("/content/news_full.csv")
true = pd.read_csv("/content/news_clean_onlly.csv")

"""**STEP 4**: Add Labels"""

fake["label"] = 0   # Fake news
true["label"] = 1   # Real news

"""**STEP 5**: Merge Dataset"""

data = pd.concat([fake, true], axis=0)
data = data.sample(frac=1).reset_index(drop=True)

"""**STEP 6**: Keep Only Useful Columns"""

data = data[["title", "text", "label"]]

"""**STEP 7**: Text Preprocessing"""

nltk.download('stopwords')
nltk.download('wordnet')

"""**Create **cleaning function"""

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

"""**STEP 8**: Apply Cleaning"""

data["content"] = data["title"] + " " + data["text"]
data["content"] = data["content"].apply(clean_text)

"""**STEP 9**: Split Data"""

X = data["content"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

"""**STEP 10**: TF-IDF Feature Extraction"""

vectorizer = TfidfVectorizer(max_features=5000)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

"""**STEP 11**: Train Model"""

model = LogisticRegression()
model.fit(X_train_vec, y_train)

"""**STEP 12**: Predictions"""

y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

"""**STEP 13**: Evaluate Model"""

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

"""**STEP 14**: Confusion Matrix"""

import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

"""**STEP 15**: Test Your Own News"""

def predict_news(news):
    cleaned = clean_text(news)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)

    if prediction[0] == 1:
        return "🟢 Real News"
    else:
        return "🔴 Fake News"

"""**STEP 16**: (Optional) Save Model"""

import joblib

joblib.dump(model, "fake_news_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
