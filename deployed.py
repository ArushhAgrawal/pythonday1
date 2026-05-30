import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import streamlit as st
import pandas as pd 
dataset= pd.read_csv('training.1600000.processed.noemoticon.csv',encoding='latin-1',header=None)
dataset.columns=["label","userid","time","query","username","tweets"]
dataset= dataset.sample(100000)
texts= dataset["tweets"].tolist()
labels=dataset["label"].replace({4: 1}).tolist()
# print(texts)
v= TfidfVectorizer()
a= v.fit_transform(texts)
#print(a)
r_train, r_test, l_train, l_test = train_test_split(a, labels, test_size=0.2)
m= LogisticRegression()
m.fit(r_train, l_train)
text= st.text_input("enter the text: ")

def prediction(text):
    v_input= v.transform([text])
    prediction = m.predict(v_input)[0]
    probability = m.predict_proba(v_input)[0][prediction]
    st.write(f"Probability: {probability}")
    return "positive" if prediction == 1 else "negative"


result = prediction(text)
st.write(f"The sentiment of the text is: {result}") 

