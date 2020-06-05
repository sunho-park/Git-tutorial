import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import  train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import MinMaxScaler, StandardScaler, robust_scale, normalize


wine  = pd.read_csv('./data/csv/winequality-white.csv', 
                        index_col=0,
                         header = 0,
                         sep=';',
                         encoding='CP949')


print(wine.head())
# print(wine.info)
print(wine.shape) #(4898, 11)
print(wine.tail())

wine = wine.values

print(type(wine))

print(wine.shape)

x_wine = wine[:, :10]
y_wine = wine[:, 10]

print(x_wine.shape)
print(y_wine.shape)

x_train, x_test, y_train, y_test = train_test_split(x_wine, y_wine, test_size=.2)

scaler = StandardScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# 모델
model = RandomForestClassifier()

# 훈련
model.fit(x_train, y_train)

# 평가 예측
score = model.score(x_test, y_test)
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
R2 = r2_score(y_test, y_predict)

print('score : ', score)
print("acc : ", acc)
print("test R2 : ", R2) 