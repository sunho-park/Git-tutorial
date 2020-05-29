from keras.datasets import cifar100
from keras.utils import np_utils
from keras.models import Sequential, Model

from keras.layers import Dense, LSTM, Conv2D
from keras.layers import Flatten, MaxPooling2D, Dropout

import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print('x_train[0] : ', x_train[0])
print('y_train[0] : \n', y_train[0])

print("y_train : \n", y_train)
print("y_test : ", y_test)
print("y_train.shape : ", y_train.shape)
print("y_test.shape : ", y_test.shape)


plt.imshow(x_train[0])
plt.show()
