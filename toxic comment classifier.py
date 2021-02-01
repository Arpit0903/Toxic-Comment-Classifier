# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 00:04:27 2019

@author: Arpit
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix

df = pd.read_csv('cleaned_train.csv')
df.head()
df_train = df.dropna()
df_test = pd.read_csv('clean_test.csv')


xtrain,xtest,ytrain,ytest = train_test_split(df_train['clean_comments'],df_train['label'].values,test_size=0.3,shuffle = True)

vectorizer = TfidfVectorizer(strip_accents='unicode',analyzer='word',ngram_range=(1,3),norm='l2')

vectorizer.fit(xtrain)
vectorizer.fit(xtest)

x_train = vectorizer.transform(xtrain)
x_test = vectorizer.transform(xtest)

nb = MultinomialNB()
nb.fit(x_train,ytrain)
y_pred = nb.predict(testit)

score = accuracy_score(ytest,y_pred)
print(score)

joblib.dump(nb,'toxicNB.pkl')
