import streamlit as st
import joblib
import re, nltk

tfidf = joblib.load('models/tfidf_vectorizer.pkl')
model = joblib.load('models/SVM_model.pkl')
le = joblib.load('models/label_encoder.pkl')

stemmer = nltk.PorterStemmer()
stop_words = set(nltk.corpus.stopwords.words('english'))

def clean(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

st.title("📰 Fake News Classifier - LIAR Dataset")
user_input = st.text_area("Paste a political statement:")

if st.button("Predict"):
    cleaned = clean(user_input)
    vec = tfidf.transform([cleaned])
    pred = model.predict(vec)
    label = le.inverse_transform(pred)
    st.success(f"Prediction: {label[0]}")
