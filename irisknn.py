# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 22:57:43 2019

@author: Lenovo
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv("iris_data.csv")


headers = ["sepal_length","sepal_width","petal_length", "petal_width","class"]
dataset.columns=headers


dataset.replace(to_replace="Iris-setosa", value=0, inplace=True)
dataset.replace(to_replace="Iris-versicolor", value=1, inplace=True)
dataset.replace(to_replace="Iris-virginica", value=2, inplace=True)

def rep(df,li):
    for i in li:
        mean=df[i].mean()
        df[i].replace(np.nan,mean,inplace=True)
rep(dataset,headers)

X = dataset.iloc[:, [0,1,2,3]].values
y = dataset.iloc[:, [4]].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25, random_state = 3)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

