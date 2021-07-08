# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 10:40:15 2021

@author: maria
"""

import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report 

x_df=pd.read_pickle("Trigram_Review")
y_df=np.load("Sentiment.npy")

X_train, X_test, y_train, y_test = train_test_split(x_df,y_df,train_size = 0.80)

pickle_in=open("MultinomialNB_hyperparameter_tuned.pickle","rb")
model_loaded=pickle.load(pickle_in)
model_loaded.fit(X_train,y_train)
print(classification_report(y_test,model_loaded.predict(X_test)))


pickle_in=open("LogisticRegression_hyperparameter_tuned.pickle","rb")
model_loaded=pickle.load(pickle_in)
model_loaded.fit(X_train,y_train)
print(classification_report(y_test,model_loaded.predict(X_test)))
