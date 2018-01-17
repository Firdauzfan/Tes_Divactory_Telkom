import csv, numpy, pandas, time
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.svm import SVC,LinearSVC

from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext, HiveContext
from pyspark.sql.functions import *
from pyspark.sql import Row
from pyspark.sql.types import *
from datetime import datetime, timedelta

start_time = time.time()

conf = (SparkConf().setAppName('bigdata'))
conf.set('spark.executor.memory','16g')

sc = SparkContext('local[6]', conf=conf)
sqlContext = SQLContext(sc)
hiveContext = HiveContext(sc)


# Baca Data

df_train = pandas.read_csv('train.csv')
df_test = pandas.read_csv('test.csv')


# Split Data Cross Validation

array = df_train.values
X = array[:,0:21]
Y = array[:,21]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
X_test = df_test
scoring = 'accuracy'


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


# Prediksi dengan Cross Validation

SVM =  LinearSVC(C=10)
SVM.fit(X_train, Y_train)
predictions = SVM.predict(X_validation)
print("\nAccuracy : {}".format(accuracy_score(Y_validation, predictions)))
print("Log Loss : {}\n".format(log_loss(Y_validation, predictions)))
print("Confusion Matrix : \n{}\n".format(confusion_matrix(Y_validation, predictions)))
print("Classification Report : \n{}".format(classification_report(Y_validation, predictions)))


print("--- %s seconds ---" % (time.time() - start_time))