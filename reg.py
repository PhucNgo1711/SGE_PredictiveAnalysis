from sklearn.datasets import load_iris
from sklearn.svm import LinearSVC
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn import preprocessing

import numpy as np
from joblib import dump

np.set_printoptions(suppress=True,
   formatter={'float_kind':'{:16.3f}'.format}, linewidth=80)

class Reg:
    filteredData = []
    rawData = None
    X_train = None
    X_test = None
    y_train = None
    y_test = None
    svmReg = None
    logReg = None
    randomForestReg = None

    def __init__(self, rawData, filteredData):
        # self.filteredData = np.array(filteredData, dtype=[('c1','i2'), ('c2','f2'), ('c3','f2'), ('c4','f2'), ('c5','f4'), 
        #                                                 ('c6','i2'), ('c7','i2'), ('c8','i2'), ('c9','i2'), ('c10','i2'),
        #                                                 ('c11','i2'), ('c12','i2'), ('c13','i2'), ('c14','i2'), ('c15','i2'),
        #                                                 ('c16','i2'), ('c17','i2'), ('c18','i2'), ('c19','i2'), ('c20','i2'),
        #                                                 ('c21','i2'), ('c22','i2'), ('c23','i2'), ('c24','i2'), ('c25','i2'),
        #                                                 ('c26','i2'), ('c27','i2'), ('c28','i2'), ('c29','i2'), ('c30','i2'), 
        #                                                 ('c31','i2'), ('c32','i2'), ('c33','i2'), ('c34','i2'), ('c35','i2'), 
        #                                                 ('c36','i2'), ('c37','i2'), ('c38','i2'), ('c39','i2'), ('c40','i2'), 
        #                                                 ('c41','i2'), ('c42','i2'), ('c43','i2')])
        self.filteredData = np.array(filteredData)
        self.rawData = rawData
        print(self.filteredData)

    def splitSet(self):

        X = self.filteredData.astype(np.float) 

        X = X[self.rawData[:, 11] == '0'] # only takes success runs

        # X = np.around(X, decimals=2)

        # Y = self.rawData[:, 11]
        # Y = Y.astype('int32')

        # print(np.around(X, decimals = 2))
        print(X)
        # # print(Y)

        # # break up dataset
        # self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, Y, test_size=0.10)

        # # scale data
        # scaler = preprocessing.StandardScaler()

        # # X_train = preprocessing.scale(X_train)`1`
        # # X_test = preprocessing.scale(X_test)
        # self.X_train = scaler.fit_transform(self.X_train)
        # self.X_test = scaler.fit_transform(self.X_test)
        # # X = scaler.fit_transform(X)
        # # print(X)
        # # print(Y)
        # print(X[:30, :]) 

        # dump(X_test[0, :], 'singleEntry.joblib')
        # dump(Y[0], 'singleEntryTest.joblib') 

        # print(X_train)
        # print(y_train)

    # random state is random seed
    # Linear SVC
    def svmReg(self):
        ##### TRAINING MODELS #####
        self.svmReg = LinearSVC(max_iter=5000, dual=False)
        self.svmReg.fit(self.X_train, self.y_train) # train using train dataset (bigger)
    
        ##### PREDICTIONS #####
        predictionSVM = self.svmReg.predict(self.X_test) # using test dataset (or input)

        # prediction == y_test # check prediction
        metricAccSVM = metrics.accuracy_score(self.y_test, predictionSVM) # prediction accuracy
        print(metricAccSVM)
        # confusion matrix
        # diagonal line = number of correct predictions
        # other columns = number of errors in that column
        metricConfMatrxSVM = metrics.confusion_matrix(self.y_test, predictionSVM)
        print(metricConfMatrxSVM)

    # Logistic Reg
    def logReg(self):
        ##### TRAINING MODELS #####
        self.logReg = linear_model.LogisticRegression(solver='lbfgs', multi_class='auto',max_iter=5000)
        self.logReg.fit(self.X_train, self.y_train)
        
        ##### PREDICTIONS #####
        predictionLog = self.logReg.predict(self.X_test)
    
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
    def rfReg(self):
        ##### TRAINING MODELS #####
        # param n for number of decision trees 
        self.randomForestReg = RandomForestRegressor(n_estimators=500)
        self.randomForestReg.fit(self.X_train, self.y_train)
   
        ##### PREDICTIONS #####
        predictionRF = self.randomForestReg.predict(self.X_test) 
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
        dump(self.svmReg, 'svmReg.joblib') 
        dump(self.logReg, 'logReg.joblib') 
        dump(self.randomForestReg, 'randomForestReg.joblib') 
        # dump(X[:30, :], 'X_test.joblib')
        # dump(Y[:30], 'Y_test.joblib')
        # dump(ar, 'ar.joblib')

