# 과제2
# Sequential 형으로 완성하시오.

# 하단에 주석으로 acc 와 loss 결과 명시하시오

import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import fashion_mnist
from keras.utils import np_utils
from keras.models import Sequential, Model

from keras.layers import Dense, LSTM, Conv2D, Input
from keras.layers import Flatten, MaxPooling2D, Dropout

# 과제1 

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print('x_train[0] : ', x_train[0])
print('y_train[0] : \n', y_train[0])

print("y_train : \n", y_train)
print("y_test : ", y_test)
print("y_train.shape : ", y_train.shape)
print("y_test.shape : ", y_test.shape)


# plt.imshow(x_train[0], 'gray')
# plt.show()

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print("y_train : ", y_train)
print("y_test : ", y_test)
print("y_train.shape : ", y_train.shape)
print("y_test.shape : ", y_test.shape)

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255

# 모델 구성
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape = (28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Dropout(0.2))

model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(512))
model.add(Dropout(0.4))
model.add(Dense(1024))
model.add(Dropout(0.6))
model.add(Dense(10, activation='softmax')) 

#loss :  0.25107481075525284
#acc :  0.9121000170707703
'''
model.add(Conv2D(32, (3, 3), activation='relu', input_shape = (28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Dropout(0.1))

model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(512))
model.add(Dropout(0.4))
model.add(Dense(1024))
model.add(Dropout(0.6))
model.add(Dense(10, activation='softmax')) #91.xx
'''
model.summary()
# 컴파일 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=15, batch_size=128, verbose=1, validation_split=1/6)

loss, acc = model.evaluate(x_test, y_test)

print("loss : ", loss)
print("acc : ", acc)



