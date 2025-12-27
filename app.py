import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

@st.cache_data
def load_data():
    data = pd.read_excel("feedback.xlsx")
    data["label"] = data["label"].str.lower()
    return data

@st.cache_resource
def train_model(data):
    X = data["text"]
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test_vec))
    return model, vectorizer, accuracy

st.set_page_config(page_title="Student Feedback Sentiment Analyzer")

st.title("🎓 Student Feedback Sentiment Analyzer")
st.write("Analyze college feedback using NLP and Machine Learning")

data = load_data()
model, vectorizer, accuracy = train_model(data)

st.success(f"Model Accuracy: {round(accuracy * 100, 2)}%")

feedback = st.text_area("Enter student feedback")

if st.button("Analyze Sentiment"):
    if feedback.strip() == "":
        st.warning("Please enter feedback text")
    else:
        vec = vectorizer.transform([feedback])
        prediction = model.predict(vec)[0]

        if prediction == "positive":
            st.success("Sentiment: POSITIVE")
        elif prediction == "neutral":
            st.info("Sentiment: NEUTRAL")
        else:
            st.error("Sentiment: NEGATIVE")

sentiment_counts = data["label"].value_counts()

fig, ax = plt.subplots()
ax.bar(sentiment_counts.index, sentiment_counts.values)
ax.set_xlabel("Sentiment")
ax.set_ylabel("Count")
ax.set_title("Student Feedback Sentiment Distribution")

st.pyplot(fig)
