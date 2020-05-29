from sklearn.datasets import load_breast_cancer

dataset = load_breast_cancer()
x = dataset.data
y = dataset.target

print('x : \n', x)
print('x.shape : ', x.shape)  #(569, 30)

print('y : ', y)
print('y.shape : ', y.shape)  # (569,)

from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv2D

model=Sequential()
model.add(10, activation='relu', input_shape=(30, ))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
