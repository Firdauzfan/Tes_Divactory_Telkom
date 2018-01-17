# -*- coding: utf-8 -*-
"""
Created on Wed Aug 02 09:05:39 2017

@author: Firdauz_Fanani
"""

import csv,numpy,pandas
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.svm import SVC,LinearSVC

#%%
#Baca Data

df_train = pandas.read_csv('train.csv')
df_test = pandas.read_csv('test.csv')

print(df_train.groupby('target').size())

#%%
#Split Data Cross Validation

array = df_train.values
X = array[:,0:21]
Y = array[:,21]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
X_test = df_test

# Test options and evaluation metric
scoring = 'accuracy'

#%%

# Pilih Algoritma
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', LinearSVC(C=10)))

# Mengevaluasi setiap model algoritma
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
 
#%%
 
# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

#%%

#Prediksi dengan Cross Validation

LDA = LinearDiscriminantAnalysis()
LDA.fit(X_train, Y_train)
predictions = LDA.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

hasil = list(LDA.predict(X_test))