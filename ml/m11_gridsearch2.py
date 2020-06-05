# RandomForest 적용
# cancer 적용

import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.utils.testing import all_estimators
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import warnings
from keras.datasets import mnist
from sklearn.ensemble import RandomForestClassifier

from sklearn.datasets import load_breast_cancer
import pandas as pd
dataset = load_breast_cancer()

x = dataset.data
y = dataset.target

# print('x : \n', x)
print('x.shape : ', x.shape)  #(569, 30)

# print('y : ', y)
print('y.shape : ', y.shape)  # (569,)


# GridSearchCV()
# RandomForestClassifier()
# parameters = {}

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=False, test_size=0.1)

parameters = {
'n_estimators': [100, 200, 300],
'criterion': ['gini', 'entropy'],
'max_depth': [5, 10, 100, 200, 500],
'min_samples_leaf': [1, 2, 5, 10],
'max_features': ["auto", "sqrt", "log2"]
}
'''

parameters = [
    {"C": [1, 10, 100, 1000], "kernel":["linear"]},
    {"C": [1, 10, 100, 1000], "kernel":["rbf"], "gamma":[0.001, 0.0001]},
    {"C": [1, 10, 100, 1000], "kernel":["sigmoid"], "gamma":[0.001, 0.0001]}
    
    ]
'''

Kfold = KFold(n_splits=5, shuffle=True)
model = GridSearchCV(RandomForestClassifier(), parameters, cv=Kfold, n_jobs=-1, verbose=1)

# 20% 검증셋을 넣겟다.
# 80% train 중 20% cross validaiton을 하겠다.

model.fit(x_train, y_train)

print("최적의 매개변수 : ", model.best_estimator_)
y_pred = model.predict(x_test)

print("최종 정답률 = ", accuracy_score(y_test, y_pred))


# https://lsjsj92.tistory.com/542