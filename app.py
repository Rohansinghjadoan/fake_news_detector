import streamlit as st
import joblib
import re
import nltk
import numpy as np
import nltk

# Download NLTK data (runs only if not downloaded)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

from sklearn.preprocessing import LabelEncoder

# Load models
tfidf = joblib.load('models/tfidf_vectorizer.pkl')
svm_model = joblib.load('models/svm_model.pkl')
xgb_model = joblib.load('models/xgboost_model.pkl')
label_encoder = joblib.load('models/label_encoder.pkl')

stemmer = nltk.PorterStemmer()
stop_words = set(nltk.corpus.stopwords.words('english'))

# ------------------ CONFIG ------------------
st.set_page_config(page_title="📰 Political Fake News Detector", page_icon="📰", layout="centered")

st.title("📰 Political Fake News Classifier")
st.markdown("**Trained on LIAR Dataset (PolitiFact)** - Classify if a political statement is *fake* or *true* with ~80% accuracy!")

# ------------------ CLEAN FUNCTION ------------------
def clean(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

# ------------------ SIDEBAR ------------------
st.sidebar.title("⚙️ Settings")
model_option = st.sidebar.selectbox("Select Classifier Model", ("SVM", "XGBoost"))

# ------------------ MAIN INPUT ------------------
user_input = st.text_area("✍️ Enter a political statement below:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a valid statement.")
    else:
        cleaned = clean(user_input)
        vec = tfidf.transform([cleaned])

        if model_option == "SVM":
            pred = svm_model.predict(vec)
            probs = svm_model.decision_function(vec)
            probs = np.max(probs)  # SVM doesn't have predict_proba
        else:
            pred = xgb_model.predict(vec)
            probs = max(xgb_model.predict_proba(vec)[0])

        label = label_encoder.inverse_transform(pred)

        st.success(f"🟣 **Prediction:** {label[0]}")
        st.progress(float(probs) if probs <= 1 else probs/100)

        st.markdown(f"**Confidence Score:** `{probs:.2f}`")

st.sidebar.markdown("— Developed by Rohan Singh Jadoan")
