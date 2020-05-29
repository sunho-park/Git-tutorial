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

# 3가지 분류니깐 원 핫 인코딩

from keras.utils import np_utils
y = np_utils.to_categorical(y)

print('y : \n', y)
print('y.shape : ', y.shape) # (150, 3)

x = x.reshape(150, 4, 1)


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
model.fit(x, y, epochs=20, batch_size=1)

# 예측, 평가

loss, acc = model.evaluate(x, y)

print("loss : ", loss)
print("acc : ", acc)
