import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from mlxtend.plotting import plot_decision_regions

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

lungcancer = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/lung-cancer/lung-cancer.data")

lungcancer.shape
lungcancer.head()


le = preprocessing.LabelEncoder()
lungcancer = lungcancer.apply(le.fit_transform)
    
X = lungcancer.drop('1', axis=1)
y = lungcancer['1']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5)


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 50)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, y_pred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(y_test,y_pred)
print("Accuracy:",result2)