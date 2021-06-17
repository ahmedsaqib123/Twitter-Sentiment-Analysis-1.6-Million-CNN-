import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image

#Heading
st.write("""
# Twitter Sentiment Analysis - 1.6Million Tweets Dataset
Information Retrival Project - Using Machine Learning & NLP
""")

#Importing Image
image = Image.open('C:/Users/Lenovo/Desktop/IR Project/sentiment.jpeg')
st.image(image,use_column_width=True)


df = pd.read_csv('C:/Users/Lenovo/Desktop/IR Project/twitter.csv',encoding='latin')
df.columns = ['sentiment','id','date','query','user_id','tweet']
df['sentiment']=df['sentiment'].replace(4,1)
st.subheader('Data Information')
st.dataframe(df.sample(20000))
st.write(df.describe())
st.subheader('Data Visualization')
st.bar_chart(df['sentiment'].value_counts())
st.area_chart(df.sample(500))

import re
import string
from string import punctuation
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

def remove_URL(text):
    url = re.compile(r"https?://\S+|www\.\S+")
    return url.sub(r"", text)

def remove_html(text):
    html = re.compile(r"<.*?>")
    return html.sub(r"", text)

text_cleaning_re = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"    

def text_cleaning(text):
    text = re.sub(text_cleaning_re, ' ', str(text).lower()).strip()
    return text

def number_cleaning(text):
    text = ''.join(c for c in text if not c.isdigit())
    return text

def remove_emoji(string):
    emoji_pattern = re.compile(
    "["
        u"\U0001F600-\U0001F64F" #emoticons
        u"\U0001F300-\U0001F5FF" #symbols & pictographs
        u"\U0001F680-\U0001F6FF" #transport & map symbols
        u"\U0001F1E0-\U0001F1FF" #FLAGS on (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub(r"", string)

def remove_punctuation(text):
    table = str.maketrans("", "", string.punctuation)
    return text.translate(table)

def stemming_words(text):
    ps=nltk.porter.PorterStemmer()
    text= ' '.join([ps.stem(word) for word in text.split()])
    return text

stop = set(stopwords.words("english"))

def remove_stopwords(text):
    text = [word.lower() for word in text.split() if word.lower() not in stop]
    
    return " ".join(text)

df["tweets"] = df['tweet'].apply(remove_URL)
df["tweets"] = df['tweet'].apply(remove_html)
df["tweets"] = df['tweet'].apply(text_cleaning)
df["tweets"] = df['tweet'].apply(number_cleaning)
df["tweets"] = df['tweet'].apply(remove_emoji)
df["tweets"] = df['tweet'].apply(remove_punctuation)
df["tweets"] = df['tweet'].apply(stemming_words)
df["tweets"] = df["tweet"].apply(remove_stopwords)


y=df['sentiment']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(df['tweets'],y,test_size=0.3,random_state=42)
print('X Train Shape: ',X_train.shape)
print('X Test Shape: ',X_test.shape)
print('Y Train Shape: ',y_train.shape)
print('Y Test Shape: ',y_test.shape)

from tensorflow.keras.preprocessing.text import Tokenizer
max_words=10000
tokenizer=Tokenizer(max_words)
tokenizer.fit_on_texts(X_train)
sequence_train=tokenizer.texts_to_sequences(X_train)
sequence_test=tokenizer.texts_to_sequences(X_test)

word2vec=tokenizer.word_index
V=len(word2vec)
print('dataset has %s number of independent tokens' %V)

from tensorflow.keras.preprocessing.sequence import pad_sequences
data_train=pad_sequences(sequence_train)
data_train.shape

T=data_train.shape[1]
data_test=pad_sequences(sequence_test,maxlen=T)
data_test.shape

from tensorflow.keras.layers import Input,Conv1D,MaxPooling1D,Dense,GlobalMaxPooling1D,Embedding
from tensorflow.keras.models import Model

D=20
i=Input((T,))
x=Embedding(V+1,D)(i)
x=Conv1D(32,3,activation='relu')(x)
x=MaxPooling1D(3)(x)
x=Conv1D(64,3,activation='relu')(x)
x=MaxPooling1D(3)(x)
x=Conv1D(128,3,activation='relu')(x)
x=GlobalMaxPooling1D()(x)
x=Dense(5,activation='softmax')(x)
model=Model(i,x)
model.summary()


model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
cnn_senti=model.fit(data_train,y_train,validation_data=(data_test,y_test),epochs=5,batch_size=256)

y_pred=model.predict(data_test)


y_pred=np.argmax(y_pred,axis=1)
score = classification_report(y_test,y_pred)
st.subheader("Model Accuracy")
st.write(str(score) +'%')


