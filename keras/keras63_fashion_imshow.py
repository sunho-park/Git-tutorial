import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import fashion_mnist

# 과제1 

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print('x_train[0] : ', x_train[0])
print('y_train[0] : \n', y_train[0])

if 9 in y_train:
    print('yes')
else:
    print('no')

print("y_train : \n", y_train)
print("y_test : ", y_test)
print("y_train.shape : ", y_train.shape)
print("y_test.shape : ", y_test.shape)


plt.imshow(x_train[0])
plt.show()
