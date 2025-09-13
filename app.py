import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 

# Initialize lemmatizer
lemma = WordNetLemmatizer()

# Preprocessing function
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(lemma.lemmatize(i))

    return " ".join(y)


tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))


st.set_page_config(page_title="SMS Spam Classifier", page_icon="📩", layout="centered")

# Custom CSS
st.markdown("""
    <style>
        body {
            background-color: #f7f9fc;
        }
        .title {
            font-size: 40px;
            font-weight: bold;
            color: #2c3e50;
            text-align: center;
        }
        .spam {
            color: white;
            background-color: #e74c3c;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            font-size: 24px;
            font-weight: bold;
        }
        .ham {
            color: white;
            background-color: #27ae60;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            font-size: 24px;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="title"> SMS Spam Classifier</div>', unsafe_allow_html=True)
st.write("### Enter a message below to check if it's Spam or Not Spam.")

# Input box
input_sms = st.text_area(" Type your message here:", height=150)

# Predict button
if st.button('Predict'):
    if input_sms.strip() == "":
        st.warning(" Please enter a message to classify.")
    else:
        # 1. preprocess
        transformed_sms = transform_text(input_sms)
        # 2. vectorize
        vector_input = tfidf.transform([transformed_sms])
        # 3. predict
        result = model.predict(vector_input)[0]
        # 4. Display
        if result == 1:
            st.markdown('<div class="spam"> Spam</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="ham"> Not Spam</div>', unsafe_allow_html=True)
