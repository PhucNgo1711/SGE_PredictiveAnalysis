from sklearn.datasets import load_iris
from sklearn.svm import LinearSVC
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import preprocessing

import numpy as np
from joblib import load
from sklearn import metrics

svmClf = load('svmClassifier.joblib')
logClf = load('logClassifier.joblib')
rf = load('randomForest.joblib')
X = load('X_test.joblib')
Y = load('Y_test.joblib')
ar = load('ar.joblib')
se = load('singleEntry.joblib').reshape(1, -1)
seTest = load('singleEntryTest.joblib').reshape(-1, 1)
predictionSVM = svmClf.predict(se)
predictionLog = logClf.predict(se)
predictionRF = rf.predict(se)
metrics.accuracy_score(seTest, predictionSVM)
metrics.accuracy_score(seTest, predictionRF)
metrics.accuracy_score(seTest, predictionLog) 

