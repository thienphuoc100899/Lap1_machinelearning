# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 10:28:08 2020

@author: HoangHieu
"""

import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer;

cancer = load_breast_cancer();
X = cancer.data;
y = cancer.target;
from sklearn.model_selection import train_test_split;

X_train , X_test, y_train, y_test = train_test_split(X,
                                                     y,
                                                     test_size = 0.2,
                                                     random_state = 0);

print(X_train.shape);
print(y_train.shape);
print("\r\n");
print(X_test.shape);
print(y_test.shape);
from sklearn.tree import DecisionTreeClassifier;

tree = DecisionTreeClassifier(criterion    =  'entropy',
                              max_depth    =  3,
                              random_state =  0 );
tree.fit(X_train, y_train)
from sklearn.tree import plot_tree;

plot_tree(tree,
          feature_names = cancer.feature_names,
          fontsize      = 7 )
from sklearn.metrics import accuracy_score;
y_pred_train = tree.predict(X_train);
print("Train Set Accuracy : ", accuracy_score(y_train, y_pred_train))
y_pred_test = tree.predict(X_test);
print("Test Set Accuracy  : ", accuracy_score(y_test, y_pred_test))
# GINI IMPURITY
tree_gin_d1 = DecisionTreeClassifier(criterion    =  'gini',
                                     max_depth    =  1,
                                     random_state =  0 );
tree_gin_d1.fit(X_train, y_train)

y_pred_train_gin_d1 = tree_gin_d1.predict(X_train);
y_pred_test_gin_d1  = tree_gin_d1.predict(X_test);


# ENTROPY IMPURITY
tree_ent_d1 = DecisionTreeClassifier(criterion    =  'entropy',
                                     max_depth    =  1,
                                     random_state =  0 );
tree_ent_d1.fit(X_train, y_train)

y_pred_train_ent_d1 = tree_ent_d1.predict(X_train);
y_pred_test_ent_d1  = tree_ent_d1.predict(X_test);
# GINI IMPURITY
tree_gin_d2 = DecisionTreeClassifier(criterion    =  'gini',
                                     max_depth    =  2,
                                     random_state =  0);
tree_gin_d2.fit(X_train, y_train)

y_pred_train_gin_d2 = tree_gin_d2.predict(X_train);
y_pred_test_gin_d2  = tree_gin_d2.predict(X_test);


# ENTROPY IMPURITY
tree_ent_d2 = DecisionTreeClassifier(criterion    =  'entropy',
                                     max_depth    =  2,
                                     random_state =  0 );
tree_ent_d2.fit(X_train, y_train)

y_pred_train_ent_d2 = tree_ent_d2.predict(X_train);
y_pred_test_ent_d2  = tree_ent_d2.predict(X_test);
# GINI IMPURITY
tree_gin_d3 = DecisionTreeClassifier(criterion    =  'gini',
                                     max_depth    =  3,
                                     random_state =  0 );
tree_gin_d3.fit(X_train, y_train)

y_pred_train_gin_d3 = tree_gin_d3.predict(X_train);
y_pred_test_gin_d3  = tree_gin_d3.predict(X_test);


# ENTROPY IMPURITY
tree_ent_d3 = DecisionTreeClassifier(criterion    =  'entropy',
                                     max_depth    =  3,
                                     random_state =  0 );
tree_ent_d3.fit(X_train, y_train)

y_pred_train_ent_d3 = tree_ent_d3.predict(X_train);
y_pred_test_ent_d3  = tree_ent_d3.predict(X_test);
print("\r\nDEPTH = 1");
print("\r\nNumber of Stump = 1");
print("\tGINI : ");
print("\t\tTrain Set Accuracy : ", accuracy_score(y_train, y_pred_train_gin_d1));
print("\t\tTest Set Accuracy  : ", accuracy_score(y_test, y_pred_test_gin_d1));
print("\tENTROPY : ");
print("\t\tTrain Set Accuracy : ", accuracy_score(y_train, y_pred_train_ent_d1));
print("\t\tTest Set Accuracy  : ", accuracy_score(y_test, y_pred_test_ent_d1));
