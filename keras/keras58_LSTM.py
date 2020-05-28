import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("x_train.shape : ", x_train.shape)  #(60000, 28, 28)
print("x_test.shape : ", x_test.shape)    #(10000, 28, 28) 
print("y_train.shape : ", y_train.shape) # (60000,) 6만개 스칼라를 가진 벡터1개
print("y_test.shape : ", y_test.shape)   # (10000,)

# 데이터 전처리

from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print("y_train : \n", y_train)
print("y_train.shape : ", y_train.shape) # (60000, 10) 
# 데이터 전처리 2. 정규화

x_train = x_train.reshape(60000, 28, 28).astype('float32')/255                                                            
x_test = x_test.reshape(10000, 28, 28).astype('float32')/255     

print("x_train.shape : ", x_train.shape)  #(60000, 28, 28)
print("x_test.shape : ", x_test.shape)    #(10000, 28, 28) 

from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, LSTM
from keras.layers import Dropout

model= Sequential()
model.add(LSTM(16, activation='relu', input_shape=(28, 28), return_sequences=False))
model.add(Dense(32, activation='relu'))
# model.add(Dense(32, activation='relu'))  

# model.add(Dense(64, activation='relu')) 
# model.add(Dropout(0.40)) # 위 레이어의 20% 제거
# model.add(Dense(128, activation='relu')) 
# model.add(Dropout(0.3))
# model.add(Conv2D(256, (5, 5))) 
# model.add(Dropout(0.3))
# model.add(Flatten())
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.6))
model.add(Dense(10, activation='softmax'))
model.summary()

# 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=15, batch_size=128, verbose=1, validation_split=1/6)

loss, acc = model.evaluate(x_test, y_test)

print("loss : ", loss)
print("acc : ", acc)


