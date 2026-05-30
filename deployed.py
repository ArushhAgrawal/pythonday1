import streamlit as st
import joblib

m = joblib.load("model.joblib")# load the model
v = joblib.load("vectorizer.joblib")# load the vectorizer

text = st.text_input("enter the text: ")

def prediction(text): #defining the pridiction function to give results
    v_input = v.transform([text])
    pred = m.predict(v_input)[0]
    probability = m.predict_proba(v_input)[0][pred]
    st.write(f"Probability: {probability}")
    return "positive" if pred == 1 else "negative"

if text:
    result = prediction(text) #outputting
    st.write(f"The sentiment of the text is: {result}")
