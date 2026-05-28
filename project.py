import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
texts = ["I absolutely loved this product",
    "This is the best thing I have ever bought",    "Amazing quality and fast delivery",
    "I am so happy with this purchase",    "Highly recommend this to everyone",
    "Excellent service and great experience",    "This product exceeded my expectations",
    "Really impressed with the quality",    "Fantastic value for the price",
    "I will definitely buy this again",    "Outstanding performance and great design",
    "Very satisfied with my order",   "This made my life so much easier",
    "Superb quality and works perfectly",    "Best purchase I have made this year",
    "This product is absolutely terrible",    "Worst purchase I have ever made",
    "Completely disappointed with the quality",    "I want my money back",
    "This is total garbage and a waste of money",    "Very poor quality and bad service",
    "I hate this product so much",    "Broke after just one day of use",    "Do not waste your money on this",    "Terrible experience from start to finish",    "Nothing works as described", "Extremely disappointed and frustrated", "i am happy ", "i am sad ", "i got beaten ", "i hate u", "the weather is bad today", "i like this weather", "the weater is sunny" , "the weather is hot outside",
    "The worst customer service I have experienced","This product is a complete scam","I regret buying this completely"]
labels = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,0, 0, 1, 1, 0, 0, 0, 0]

# print(texts)
v= TfidfVectorizer()
a= v.fit_transform(texts)
#print(a)
r_train, r_test, l_train, l_test = train_test_split(a, labels, test_size=0.2)
m= LogisticRegression()
m.fit(r_train, l_train)

text= input("enter the text: ")

def prediction(text):
    v_input= v.transform([text])
    prediction = m.predict(v_input)[0]
    probability = m.predict_proba(v_input)[0][prediction]
    print(probability)
    return "positive" if prediction == 1 else "negative"


result = prediction(text)
print(result)



