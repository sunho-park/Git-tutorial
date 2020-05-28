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

# print(x_train[0])
# plt.imshow(x_train[0], 'gray')
# plt.imshow(x_train[0])
# plt.show()


# 데이터 전처리

from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print("y_train : \n", y_train)
print("y_train.shape : ", y_train.shape) # (60000, 10) https://ko.d2l.ai/chapter_crashcourse/linear-algebra.html (텐서 개념)

# 데이터 전처리 2. 정규화

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255                                                            
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255     

# 0(흰색) 255(완전진한검정) 
# reshape로 4차원 why cnn에 집어넣으려고 // x 각 픽셀마다 정수형태 0~255가 들어가 있음 min max 0~1 
# 255로 나누는 이유, 정규화                 y 는 0~9 까지
# x_train = x_train / 255   # (x - 최소) / (최대 - 최소)

from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, LSTM, Dropout
from keras.layers import Input

input1 = Input(shape=(28,28,1))
dense1_1 = Conv2D(32, (2, 2))(input1)
dense1_2 = Conv2D(64, (2, 2))(dense1_1) 
dense1_2 = MaxPooling2D(pool_size=(2, 2))(dense1_1)
dense1_2 = Dropout(0.20)(dense1_1)

dense1_3 = Conv2D(128, (2, 2))(dense1_2)
dense1_3 = MaxPooling2D(pool_size=(2, 2))(dense1_2)
# model.add(Dropout(0.3))
# model.add(Conv2D(256, (5, 5))) 
# model.add(Dropout(0.3))


output1 = Conv2D(256, (2, 2))(dense1_3)
output1 = Dropout(0.5)(dense1_3)
output1 = Flatten()(output1)
output1_1 = Dense(10, activation='softmax')(output1)

model = Model(inputs=input1, outputs= output1_1)
model.summary()

# 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=5, batch_size=128, verbose=1, validation_split=1/6)

loss, acc = model.evaluate(x_test, y_test)

print("loss : ", loss)
print("acc : ", acc)


