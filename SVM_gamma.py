# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 10:16:27 2018

@author: sim
"""

import pandas as pd
#from MLModel import MLModel

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from matplotlib import pyplot as plt
import numpy as np

droplist = set()
RANDOM_STATE = 42
N_FOLDS = 10

def parseRecord(v, l):
    return [l.index(i) for i in v]

def csv_to_data():
    # load the CSV file
    data = pd.read_csv('./crx.csv', header=None)
    data = data.sample(frac=1).reset_index(drop=True)

    #drop if missing values
    global droplist
    for index, row in data.iterrows():
        for i in range(0,16):
            if row[i] is '?':
                droplist.add(index)

    data = data.drop(list(droplist))

    # Pluses and minuses
    classification = data[15]

    # Numerical columns numbers: 1, 2, 7, 10, 13, 14
    data[0] = parseRecord(data[0], ['a', 'b'])
    data[3] = parseRecord(data[3], ['u', 'y', 'l', 't'])
    data[4] = parseRecord(data[4], ['g', 'p', 'gg'])
    data[5] = parseRecord(data[5], ['c', 'd', 'cc', 'i', 'j', 'k', 'm', 'r', 'q', 'w', 'x', 'e', 'aa', 'ff'])
    data[6] = parseRecord(data[6], ['v', 'h', 'bb', 'j', 'n', 'z', 'dd', 'ff', 'o'])
    data[8] = parseRecord(data[8], ['t', 'f'])
    data[9] = parseRecord(data[9], ['t', 'f'])
    data[11] = parseRecord(data[11], ['t', 'f'])
    data[12] = parseRecord(data[12], ['g', 'p', 's'])

    features = data.drop(15, 1)

    #turn +, - into 1, 0

    s = pd.Series()
    for num, sign in classification.iteritems():
        if sign is '+':
            s.set_value(num, 1)
        else:
            s.set_value(num, 0)
    classification = s
    return features, classification
    
if __name__ == "__main__":

    x = []
    y = []
    features, classification = csv_to_data()
    features = features.reset_index(drop=True)
    classification = classification.reset_index(drop=True)
    # k-Fold Set for Cross-validation
    cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    scores = pd.DataFrame(columns=['accuracy', 'precision', 'recall', 'f1'])
    
    cnt = 0
    a = 0.01
    b= 10
    for number in range(1,100) :
        for train_index, test_index in cv.split(features, classification):

            features_train, features_test = features.loc[train_index, :], features.loc[test_index, :]
            classification_train, classification_test = classification[train_index], classification[test_index]

            ##RandomForestClass parameter 정하기
            #clf = RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=8)
            #clf = DecisionTreeClassifier(criterion='entropy',splitter='best', max_depth=5, min_samples_leaf=3)
            #clf = SVC(kernel="poly",gamma = 'auto', degree = 2 , C=100 , epsilon = 0.1 )
            #clf = SVC(gamma= a)
            #clf = LinearSVC(C=1, max_iter = 2000)
            #clf = LogisticRegression(solver = 'liblinear', C=10) #C값을 올리면 overfitting
            #clf = RandomForestClassifier(n_estimators = number) #n_estimator 트리의 갯수
            #clf = RandomForestClassifier(n_estimators = 1000, max_depth = number,n_jobs = -1)
            #clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3), n_estimators = 200, learning_rate = 0.5)
            # clf = MLPClassifier(activation='logistic', alpha=1e-5, hidden_layer_sizes=(number*5, number*4, number*3, number*2,number), random_state=1,  solver='adam')
            clf = SVC(gamma=a, C = 10-a)
            clf.fit(features_train, classification_train)

            classification_true, classification_pred = classification_test.values, clf.predict(features_test)

            prs_tmp1 = accuracy_score(classification_true, classification_pred)
            prs_tmp2 = precision_recall_fscore_support(classification_true, classification_pred, average='binary')

            scores.at[cnt, 'accuracy'] = prs_tmp1
            scores.at[cnt, 'precision'] = prs_tmp2[0]
            scores.at[cnt, 'recall'] = prs_tmp2[1]
            scores.at[cnt, 'f1'] = prs_tmp2[2]
            cnt+=1

        a = a+0.1
        scores.at['mean', 'accuracy']=scores.accuracy.mean()
        scores.at['mean', 'precision']=scores.precision.mean()
        scores.at['mean', 'recall']=scores.recall.mean()
        scores.at['mean', 'f1']=scores.f1.mean()

        print(number ,">>", scores.accuracy.mean())


        x.append(a)
        y.append(scores.accuracy.mean())


plt.plot(x,y)
plt.savefig("SVM_gamma.png")
plt.xlabel('gamma')
plt.ylabel('Accuracy')
plt.title('SVM_gamma')
plt.show()



    
scores.to_excel('results.xlsx', 'Sheet1')