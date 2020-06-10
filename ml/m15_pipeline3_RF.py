# RandomizedSearchCV + Pipeline
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier


# 1. 데이터
iris = load_iris()

x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle=True, random_state=43)

# 그리드 / 랜덤 서치에서 사용할 매개 변수

'''
parameters = [
    {"svm__C" :[1, 10, 100, 1000], "svm__kernel" :['linear']},
    {"svm__C" :[1, 10, 100], "svm__kernel" :['rbf'], 'svm__gamma':[0.001, 0.0001]},
    {"svm__C" :[1, 100, 1000], "svm__kernel" :['sigmoid'], 'svm__gamma':[0.001, 0.0001]}
]
'''
'''
parameters = [
    {"C" :[1, 10, 100, 1000], "kernel" :['linear']},
    {"C" :[1, 10, 100], "kernel" :['rbf'], 'gamma':[0.001, 0.0001]},
    {"C" :[1, 100, 1000], "kernel" :['sigmoid'], 'gamma':[0.001, 0.0001]}
]
'''
parameters = [
    {"rf__n_jobs" :[1], "rf__n_estimators" : [100, 200], "rf__criterion" : ['gini', 'entropy']},
    {"rf__min_samples_leaf" : [1, 2], "rf__max_features" : [1, 2]}
]

# 2. 모델
# model = SVC()

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

pipe = Pipeline([("scaler", MinMaxScaler()), ('rf', RandomForestClassifier())])
# pipe = make_pipeline(MinMaxScaler(), RandomForestClassifier())

model = RandomizedSearchCV(pipe, parameters, cv=5)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
acc = model.score(x_test, y_test)

print("최적의 매개변수 = ", model.best_estimator_)
print("최적의 매개변수 = ", model.best_params_)
print("acc : ", acc)

import sklearn as sk
print("sklearn :", sk.__version__)


# pipe.fit(x_train, y_train)
# print("acc : ", pipe.score(x_test, y_test))

