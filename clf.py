from sklearn.datasets import load_iris
from sklearn.svm import LinearSVC
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import preprocessing

import numpy as np
from joblib import dump

class Clf:
    filteredData = []
    rawData = None
    X_train = None
    X_test = None
    y_train = None
    y_test = None
    svmClf = None
    logClf = None
    randomForestClf = None

    def __init__(self, rawData, filteredData):
        self.filteredData = np.array(filteredData)
        self.rawData = rawData
        # print(self.filteredData)

    def splitSet(self):
        X = self.filteredData.astype('float32')
        Y = self.rawData[:, 11]
        Y = Y.astype('int32')

        # print(X)
        # print(Y)

        # break up dataset
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, Y, test_size=0.10)

        # scale data
        scaler = preprocessing.StandardScaler()

        # X_train = preprocessing.scale(X_train)`1`
        # X_test = preprocessing.scale(X_test)
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.fit_transform(self.X_test)
        # X = scaler.fit_transform(X)
        # print(X)
        # print(Y)
        # print(X[:30, :]) 

        # dump(X_test[0, :], 'singleEntry.joblib')
        # dump(Y[0], 'singleEntryTest.joblib') 

        # print(X_train)
        # print(y_train)

     # random state is random seed
    # Linear SVC
    def svmClf(self):
        ##### TRAINING MODELS #####
        self.svmClf = LinearSVC(max_iter=5000, dual=False)
        self.svmClf.fit(self.X_train, self.y_train) # train using train dataset (bigger)
    
        ##### PREDICTIONS #####
        predictionSVM = self.svmClf.predict(self.X_test) # using test dataset (or input)

        # prediction == y_test # check prediction
        metricAccSVM = metrics.accuracy_score(self.y_test, predictionSVM) # prediction accuracy
        print(metricAccSVM)
        # confusion matrix
        # diagonal line = number of correct predictions
        # other columns = number of errors in that column
        metricConfMatrxSVM = metrics.confusion_matrix(self.y_test, predictionSVM)
        print(metricConfMatrxSVM)

    # Logistic Reg
    def logClf(self):
        ##### TRAINING MODELS #####
        self.logClf = linear_model.LogisticRegression(solver='lbfgs', multi_class='auto',max_iter=5000)
        self.logClf.fit(self.X_train, self.y_train)
        
        ##### PREDICTIONS #####
        predictionLog = self.logClf.predict(self.X_test)
    
        # prediction == y_test # check prediction
        metricAccLog = metrics.accuracy_score(self.y_test, predictionLog)
        # metricAccLog = metrics.accuracy_score(Y, predictionLog)
        print(metricAccLog)
        # confusion matrix
        # diagonal line = number of correct predictions
        # other columns = number of errors in that column
        metricConfMatrxLog= metrics.confusion_matrix(self.y_test, predictionLog)
        # metricConfMatrxLog= metrics.confusion_matrix(Y, predictionLog)
        print(metricConfMatrxLog)

    # Random Forest
    def rfClf(self):
        ##### TRAINING MODELS #####
        # param n for number of decision trees 
        self.randomForestClf = RandomForestClf(n_estimators=500)
        self.randomForestClf.fit(self.X_train, self.y_train)
   
        ##### PREDICTIONS #####
        predictionRF = self.randomForestClf.predict(self.X_test) 
        # prediction == y_test # check prediction
        metricAccRF = metrics.accuracy_score(self.y_test, predictionRF)
        # metricAccRF = metrics.accuracy_score(Y, predictionRF)
        print(metricAccRF)
        # confusion matrix
        # diagonal line = number of correct predictions
        # other columns = number of errors in that column
        metricConfMatrxRF= metrics.confusion_matrix(self.y_test, predictionRF)
        # metricConfMatrxRF= metrics.confusion_matrix(Y, predictionRF)
        print(metricConfMatrxRF)
    
    def releaseModel(self):
        # dump(self.svmClf, 'svmClf.joblib') 
        # dump(self.logClf, 'logClf.joblib') 
        # dump(self.randomForestClf, 'randomForestClf.joblib') 
        # dump(X[:30, :], 'X_test.joblib')
        # dump(Y[:30], 'Y_test.joblib')
        # dump(ar, 'ar.joblib')
        return

