import numpy as np
import matplotlib.pyplot as plt 
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()


print("x_train.shape : ", x_train.shape)  #(60000, 28, 28)
print("x_train : \n", x_train)
print("x_train[0]: ", x_train[0])
print('y_train : ', y_train[0])

print("x_test.shape : ", x_test.shape)    #(10000, 28, 28) 

print("y_train.shape : ", y_train.shape) # (60000,) 6만개 스칼라를 가진 벡터1개
print("y_test.shape : ", y_test.shape)   # (10000,)

print("x_train[0].shape : ", x_train[0].shape) #(28, 28)

from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255

print("y_train : \n", y_train)
print("y_train.shape : ", y_train.shape)

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

model = Sequential()
model.add(Conv2D(32, (1, 1), input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3)))
model.add(Conv2D(128, (3, 3)))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

# 컴파일 훈련

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=7, batch_size=128, verbose=1, validation_split=0.2)

loss, acc = model.evaluate(x_test, y_test)

print("loss : ", loss)
print("acc : ", acc)

