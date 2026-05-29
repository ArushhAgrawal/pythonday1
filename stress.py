from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
dataset= pd.read_csv('dreaddittest.csv',encoding='latin-1',header=0)
dataset.sample(700)
texts=dataset["text"].tolist()
labels=dataset["label"].tolist()
v= TfidfVectorizer()
a= v.fit_transform(texts)
r_train, r_test, l_train, l_test = train_test_split(a, labels,test_size=0.2, random_state=69)
m= LogisticRegression(max_iter=1000)
m.fit(r_train, l_train)
while True:
   text= input("enter the text: ")
   if text.lower() == "exit":
       break

   def prediction(text):
    v_input=v.transform([text])
    prediction=m.predict(v_input)[0]
    probability=m.predict_proba(v_input)[0][prediction]
    print(probability)
    if prediction == 1:
        return "highly stressed" if probability > 0.69 else "stressed"
    else:
       return "not stressed"
   
   result= prediction(text)
   print(result)