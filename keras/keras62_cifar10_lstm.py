from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential, Model

from keras.layers import Dense, LSTM, Conv2D, Input
from keras.layers import Flatten, MaxPooling2D, Dropout

import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train[0])
print('y_train[0] : ', y_train[3])

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# plt.imshow(x_train[0])
# plt.show()

from keras.utils import np_utils

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)


print("y_train : ", y_train)
print("y_test : ", y_test)

x_train = x_train.reshape(50000, 64, 48).astype('float32')/255
x_test = x_test.reshape(10000, 64, 48).astype('float32')/255

input1 = Input(shape=(64, 48))
dense1 = LSTM(8, activation='relu', return_sequences=False)(input1)
dense1 = Dense(16, activation='relu')(dense1)  # 3차원에서 2차원

output1 = Dense(32, activation='relu')(dense1)
output1_1 = Dense(10, activation='softmax')(output1)

model = Model(inputs= input1, outputs = output1_1)

model.summary()

# 컴파일 훈련

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=5, batch_size=64, verbose=1 ,validation_split=0.2)

loss, acc = model.evaluate(x_test, y_test)

print("loss : ", loss)
print("acc : ", acc)
