from sklearn.datasets import load_iris
import numpy as np

from keras.models import Sequential, Model
from keras.layers import Dense, LSTM

dataset = load_iris()
x = dataset.data
y = dataset.target


print('x : \n', x)
print('x.shape : ', x.shape) #(150, 4)
print('y : ', y)
print('y.shape : ', y.shape) #(150, ) - > (150, 1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=False, test_size=0.1)

# 3 분류니깐 원핫인코딩
from keras.utils import np_utils

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print('y_train : \n', y_train)
print('y_test : ', y_test)


print('y_train.shape : ', y_train.shape)
print('y_test.shape : ', y_test.shape)

x_train = x_train.reshape(135, 4, 1) 
x_test = x_test.reshape(15, 4, 1)


# 모델 구성
model = Sequential()

model.add(LSTM(10, activation='relu', input_shape=(4, 1), return_sequences=False))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax')) 

model.summary()

# 컴파일, 훈련

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=20, batch_size=1, verbose=1)

# 예측, 평가

loss, acc = model.evaluate(x_test, y_test)

print("loss : ", loss)
print("acc : ", acc)
