import numpy as np
import pandas as pd

kospi200 = np.load('./data/kospi200/data/kospi200.npy')
samsung = np.load('./data/kospi200/data/samsung.npy')

print('kospi200 : \n', kospi200)
print('samsung : \n', samsung)
print('kospi200.shape : ', kospi200.shape)
print('samsung.shape : ', samsung.shape)

def split_xy5(dataset, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column

        if y_end_number > len(dataset):
            break
        
        tmp_x = dataset[i:x_end_number, :]
        tmp_y = dataset[x_end_number:y_end_number, 3]

        x.append(tmp_x)
        y.append(tmp_y)

    return np.array(x), np.array(y)
x, y = split_xy5(samsung, 5, 1)

print('x : ', x)
print('y : ', y)

print('x[0, :] =\n', x[0, :], "\n", y[0])  

print(x.shape)  #(421, 5, 5)   
print(y.shape)  #(421, 1)

# 데이터셋 나누기

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, test_size=0.3)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

x_train = x_train.reshape(294, 25)
x_test = x_test.reshape(127, 25)

print(x_train.shape)
print(x_test.shape)

