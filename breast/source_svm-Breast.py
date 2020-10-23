# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 10:37:20 2020

@author: HoangHieu
"""

import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import grid_search, svm
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
#import the raw data
raw_data = pd.read_csv("breast-cancer-wisconsin.data", 
                       names = ["id",  "Clump Thickness", "Uniformity of Cell Size","Uniformity of Cell Shape", 
                                  "Marginal Adhesion", "Single Epithelial Cell Size","Bare Nuclei", "Bland Chromatin",
                                  "Normal Nucleoli", "Mitoses", "Class"])
print(raw_data.shape)
raw_data.head()
# drop ID and Class columns
raw_data2 = raw_data.drop(['id','Class'], axis=1)

# normalize the data to have a mean of 0 and std deviation of 1 (standard normal distribution)
# normalize by subtracting raw scores from mean and dividing by std deviation (z-score)
norm_data = (raw_data2 - np.mean(raw_data2)) / np.std(raw_data2)
norm_data.head()

# map class variable to 1's (malignant) and 0's (benign)
norm_data['Class'] = raw_data['Class'].map({4:1, 2:0})
norm_data.head()
# divide normalized data into features and labels
features = norm_data.drop('Class', axis=1)
labels = norm_data['Class']
print(labels.head())
features.head()

# split data into training and test features and labels using 30% of data as validation/test set
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=.3)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
# create function svc_param_selection that optimizes combination of degree and C hyperparameters using an SVC with a
# polynomial kernel
# this function was used in lecture 6 of UC Berkeley's Machine Learning Decal

def svc_param_selection(X, y, nfolds):
    """ When using a SVM with a polynomial kernel there are two hyperparameters to tune. The value of C and 
    the degree of the polynomial, d. This function, svc_param_selection will find the optimal pair of (C, degree)
    that gives the best results on a test set using sklearn's GridSearchCV (cross validation) method."""
    #the slack penalty hyperparameter
    Cs = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
    #degrees of polynomial kernel of svc
    degrees = [1, 2, 3, 4, 5]
    #initialize the paremeter grid as dictionary
    param_grid = {'C': Cs, 'degree' : degrees}
    #initialize search for best parameters using input nfold cross validation
    search = grid_search.GridSearchCV(svm.SVC(kernel='poly'), param_grid, cv=nfolds)
    #fit the search object to input training data
    search.fit(X, y)
    #return the best parameters
    search.best_params_
    return search.best_params_
svc_param_selection(X_train, y_train, 10)

final_svc_poly = svm.SVC(C=.1, degree=1, kernel='poly')
final_svc_poly.fit(X_train, y_train)
final_svc_poly.score(X_test, y_test)
Cs = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
degrees = [1, 2, 3, 4, 5]

train_acc = []
test_acc = []

for d in degrees:
    for c in Cs:
        #print("C = ", c, ", degree = ", d)
        svc = svm.SVC(C=c, degree=d, kernel='poly')
        svc.fit(X_train, y_train)
        #print(svc.score(X_train, y_train))
        train_acc.append((svc.score(X_train, y_train)))
        #print((svc.score(X_test, y_test)))
        test_acc.append((svc.score(X_test, y_test)))
#print(max(test_acc))
# load test accuracies into 2D numpy array
acc_img = np.array(test_acc).reshape(len(degrees), len(Cs))

# plot heatmap of accuracies
plt.imshow(acc_img, cmap=plt.cm.hot_r)
plt.colorbar()
plt.show()
plt.title('Test Set Accuracies')
plt.xticks(np.arange(len(Cs)), Cs, rotation=60)
plt.yticks(np.arange(len(degrees)), degrees)
plt.xlabel('Misclassification Penalty Values (C)')
plt.ylabel('Degree of Polynomial Kernel (d)');
final_svc_poly = svm.SVC(C=.1, degree=1, kernel='poly')
final_svc_poly.fit(X_train, y_train)
final_svc_poly.score(X_test, y_test)
from sklearn.metrics import confusion_matrix

confusion_mat = confusion_matrix(y_test, final_svc_poly.predict(X_test))
confusion_mat