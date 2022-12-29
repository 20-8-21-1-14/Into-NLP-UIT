import string
import time
import streamlit as st
import pickle
import nltk
import re
import pandas as pd
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

# punct%
def count_punct(text):
    count = sum([1 for char in text if char in string.punctuation])
    return round(count/(len(text) - text.count(" ")), 3) * 100

# Symbol Currency
def currency(x):
  currency_symbols = ['€', '$', '¥', '£', '₹']
  for i in currency_symbols:
    if i in x:
      return 1
  return 0

# Contain Number
def numbers(x):
  for i in x:
    if ord(i)>=48 and ord(i)<=57:
      return 1
  return 0


def prcss_input_frm_user(data):
    data['body_len'] = data.v2.apply(lambda x: len(x) - x.count(" "))
    data['word_count'] = data.v2.apply(lambda x: len(x.split()))

    data['punct_rate'] = data.v2.apply(lambda x: count_punct(x))
    data['contains_currency_symbol'] = data.v2.apply(currency)

    data['contains_number'] = data.v2.apply(numbers)
    return data

def main():
    st.title('SMS Spam Classifier')
    # tf_idf_processor_1 = pickle.load(open('preprocess/vectorizer_lemm.pkl', 'rb'))
    model_lemm =  pickle.load(open('model/model_rf_lemm.pkl', 'rb'))
    try:
        input_sms = st.text_area("Enter the message")
        if st.button('Send'):
            msg = pd.DataFrame({"v2": input_sms}, index=[0])
            print("head", msg.head())
            prcss_input_frm_user(msg)
            print("head", msg)
            res = model_lemm.predict(msg)

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