from sklearn.datasets import load_diabetes
import numpy as np

from sklearn import model_selection
from sklearn import metrics
import pandas as pd

dataset = load_diabetes()
x = dataset.data
y = dataset.target

print('x : \n', x)
print('x.shape : ', x.shape) # (442, 10)
print('y : ', y)
print('y.shape : ', y.shape) #(442,)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=False, test_size=0.1)


print("x_train.shape : ", x_train.shape)
print("x_test.shape : ", x_test.shape)

print("y_train.shape : ", y_train.shape)          
print("y_test.shape : ", y_test.shape)   

x_train = x_train.reshape(397, 10, 1, 1)
x_test = x_test.reshape(45, 10, 1, 1)


from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv2D, Flatten

model = Sequential()

model.add(Conv2D(10, (1, 1), input_shape = (10, 1, 1), activation='relu'))
model.add(Conv2D(50, (1, 1), activation='relu'))
model.add(Conv2D(50, (1, 1), activation='relu'))
model.add(Conv2D(50, (1, 1), activation='relu'))
model.add(Conv2D(50, (1, 1), activation='relu'))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(1))

# 컴파일, 훈련

model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=20, batch_size=1, verbose=1, validation_split=0.1)

# 훈련 평가

loss, mse = model.evaluate(x_test, y_test)

print("loss : ", loss)
print("mse : ", mse)