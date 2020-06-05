import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import  train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import MinMaxScaler, StandardScaler, robust_scale, normalize
from keras.utils import np_utils
from keras.layers import Dense
from keras.models import Sequential

wine  = pd.read_csv('./data/csv/winequality-white.csv', 
                        index_col=0,
                         header = 0,
                         sep=';',
                         encoding='CP949')


# print(wine.head())
# print(wine.info)
print(wine.shape)  #(4898, 11)
# print(wine.tail())


wine = wine.values

print(type(wine))

print('wine.shape : ', wine.shape) # (4898, 11)

# x, y 자르기
x_wine = wine[:, :10]
y_wine = wine[:, 10]

print(x_wine.shape)  # x (4898, 10)
print(y_wine.shape)  # y (4898,)

print('y_wine : ', y_wine)

# 다중분류이므로 Y 에 대해서 원 핫 인코딩 
y_wine = np_utils.to_categorical(y_wine)
print('y_wine.shape : ', y_wine.shape) # (4898, 10)

y_wine = y_wine[:, 1:]
print('y_wine.shape : ', y_wine.shape) #(4898, 9)

# 데이터셋 train, test 분리
x_train, x_test, y_train, y_test = train_test_split(x_wine, y_wine, test_size=.2)

print('y_train.shape : ', y_train.shape)    #(3918, 9)
print('y_test.shape : ', y_test.shape)      #(980, 9)

# 표준화
scaler = StandardScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# SHAPE 확인
print('x_train.shape : ', x_train.shape)     # (3918, 10)
print('x_test.shape : ', x_test.shape)       # (980, 10)

print('y_train.shape : ', y_train.shape)    #(3918, 9)
print('y_test.shape : ', y_test.shape)      # (980, 9)


# 모델
model = Sequential()
model.add(Dense(100, input_dim=10, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(9, activation='softmax'))


 
# 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=100, batch_size=128, verbose=1, validation_split=0.3 )


# 평가 예측
loss, acc = model.evaluate(x_test, y_test)

print("loss : ", loss)
print("acc : ", acc)
