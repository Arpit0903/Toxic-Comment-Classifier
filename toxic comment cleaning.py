# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 22:27:17 2019

@author: Arpit
"""
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import re

df = pd.read_csv('train.csv')
df1 = pd.read_csv('test.csv')

comments = []
for i in range(len(df1['comment_text'])):
    comments.append(i)

#DATA CLEANING  
def remove_punct(text):
    text  = "".join([char for char in text if char not in string.punctuation])
    text = re.sub('[0-9]+', '', text)
    return text

df1['comment_punct'] = df1['comment_text'].apply(lambda x: remove_punct(x))

def tokenize_comments(text):
    text = re.split('\W+',text)
    return text

df1['comment_tokenized'] = df1['comment_punct'].apply(lambda x: tokenize_comments(x.lower()))

stw = stopwords.words('english')
stw.extend(['yr', 'year', 'woman', 'man', 'girl','boy','one', 'two', 'sixteen', 'yearold', 
            'fu', 'weeks', 'week','treatment','associated', 'patients', 'may','day', 
            'case','old'])

def remove_stopwords(text):
    text = [word for word in text if word not in stw]
    return text

df1['comment_nonstop'] = df1['comment_tokenized'].apply(lambda x: remove_stopwords(x))

lemmatizer = WordNetLemmatizer()
def lemmatize_comments(text):
    text = [lemmatizer.lemmatize(word) for word in text]
    return text

df1['comment_lemmatized'] = df1['comment_nonstop'].apply(lambda x: lemmatize_comments(x))

clean_comments = []
for i in df1['comment_lemmatized']:
    clean_comments.append(i)

for i in range(len(clean_comments)):
    clean_comments[i]= ' '.join(word for word in clean_comments[i])
    
df1['clean_comments'] = clean_comments

#vectorizer = TfidfVectorizer()
#matrix = vectorizer.fit_transform(clean_comments)
#matrix = np.asarray(matrix)

columns = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
df_dist = df[columns].sum()\
                     .to_frame()\
                     .rename(columns={0:'count'})\
                     .sort_values('count')
df_dist.sort_values('count',ascending=False)

label = []
label_toxic = df['toxic']
label_obscene = df['obscene']
label_sevt = df['severe_toxic']
label_insult = df['insult']
label_iden = df['identity_hate']
label_threat = df['threat']

for i in range(len(df)):
    if(label_toxic[i]== 0 & label_sevt[i]== 0 & label_obscene[i]== 0 & label_insult[i]==0 & label_insult[i]==0 & label_iden[i]==0):
        label.append(None)
    elif(label_toxic[i]== 1 & label_sevt[i]== 1 & label_obscene[i]== 1 & label_insult[i]==1 & label_insult[i]==1 & label_iden[i]==1):
        label.append('obscene')
    elif(label_toxic[i] == 1 & label_sevt[i] ==1):
        label.append('severe toxic')
    elif(label_toxic[i]==1 & label_obscene[i]==1):
        label.append('obscene')
    elif(label_toxic[i]==1 & label_insult[i]==1):
        label.append('insult')
    elif(label_toxic[i]==1 & label_threat[i]==1):
        label.append('threat')
    elif(label_toxic[i]==1 & label_iden[i]==1):
        label.append('identity hate')
    elif(label_threat[i]==1 & label_obscene[i]==1):
        label.append('obscene')
    elif(label_iden[i]==1 & label_obscene[i]==1):
        label.append('obscene')
    elif(label_threat[i]==1 & label_iden[i]==1):
        label.append('identity hate')
    elif(label_toxic[i] == 1):
        label.append('toxic')
    elif(label_obscene[i] == 1):
        label.append('obscene')
    elif(label_sevt[i] == 1):
        label.append('severe toxic')
    elif(label_insult[i] == 1):
        label.append('insult')
    elif(label_threat[i] == 1):
        label.append('threat')
    elif(label_iden[i] == 1):
        label.append('identity hate')
    else:
        label.append('toxic')
        
df['label'] = label

