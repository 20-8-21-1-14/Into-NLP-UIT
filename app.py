import time
import streamlit as st
import pickle
import nltk
import numpy as np

sent_success_message = 'Message sent!'
warning_message = 'You entered a spam message!'
error_message = 'Something wrong!'
stopwords = nltk.corpus.stopwords.words('english')

Stemmer = nltk.PorterStemmer()
Lemmatizer = nltk.WordNetLemmatizer()

res = 0

def process_text_with_Lemmatizer(data_set):
    data_set = "".join([word.lower() for word in data_set if word not in string.punctuation])
    tokens = re.findall('\S+', data_set)
    text = [Lemmatizer.lemmatize(word) for word in tokens if word not in stopwords]
    return text


def main():
    st.title('SMS Spam Classifier')
    tab1, tab2 = st.tabs(["Lemmatizer", "Stemmer"])
    with tab1:
        tab1.subheader("Lemmatizer method")
        tf_idf_processor_1 = pickle.load(open('preprocess/vectorizer_lemm.pkl', 'rb'))
        model_lemm =  pickle.load(open('model/model_rf_lemm.pkl', 'rb'))

        try:
            input_sms = st.text_area("Enter the message",key=1)
            if st.button('Send', key=11):
                msg = tf_idf_processor_1.transform([str(input_sms)]).toarray()
                res = model_lemm.predict(msg)

                with st.spinner('Wait for it...'):
                    time.sleep(1)

                if res[0]==1:
                    st.warning(warning_message, icon="⚠️")
                else:
                    st.success(sent_success_message, icon="✅")
        except Exception as e:
            st.error(error_message, icon="🚨")

    with tab2:
        tab2.subheader("Stemmer method")
        tf_idf_processor_2 = pickle.load(open('preprocess/vectorizer_stemm.pkl', 'rb'))
        model_stemm =  pickle.load(open('model/model_rf_stemm.pkl', 'rb'))

        try:
            input_sms = st.text_area("Enter the message",key=2)
            if st.button('Send',key=22):
                msg = tf_idf_processor_2.transform([str(input_sms)]).toarray()
                res = model_stemm.predict(msg)

                with st.spinner('Wait for it...'):
                    time.sleep(1)

                if res[0]==1:
                    st.warning(warning_message, icon="⚠️")
                else:
                    st.success(sent_success_message, icon="✅")
        except Exception as e:
            print(e)
            st.error(error_message, icon="🚨")


if __name__=='__main__':
    main()