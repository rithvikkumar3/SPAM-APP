
import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    
    words = []
    for token in tokens:
        if token.isalnum():
            words.append(token)
    
    filtered_words = []
    for word in words:
        if word not in stopwords.words('english') and word not in string.punctuation:
            filtered_words.append(word)
            
    stemmed_words = []
    for word in filtered_words:
        stemmed_words.append(ps.stem(word))
    
    return " ".join(stemmed_words)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email/SMS Spam Classifier")
st.markdown("---")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    with st.spinner('Analyzing the message...'):
        # 1. preprocess
        transformed_sms = transform_text(input_sms)
        # 2. vectorize
        vector_input = tfidf.transform([transformed_sms])
        # 3. predict
        result = model.predict(vector_input)[0]
    
    st.subheader("Prediction Result:")
    if result == 1:
        st.error("This message is classified as Spam.")
    else:
        st.success("This message is classified as Not Spam.")

st.sidebar.markdown("### Tips:")
st.sidebar.write("1. Enter a message in the text area.")
st.sidebar.write("2. Click the 'Predict' button to see the classification result.")
