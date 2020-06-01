# 95번을 불러와서 모델을 완성하시오.
import numpy as np

iris = np.load('./data/csv/iris.npy')

print(iris)

x = iris[:, :4]
print(x)

y = iris[:, 4:]
print(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, test_size=0.1)

# 원 핫 인코딩
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print('y_train : \n', y_train)
print('y_test : ', y_test)

print('y_train.shape : ', y_train.shape)
print('y_test.shape : ', y_test.shape)
print('x_train.shape : ', x_train.shape)
print('x_test.shape : ', x_test.shape)

# 모델 구성
from keras.models import Sequential
from keras.layers import Dense 

model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(4, )))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax'))

# 컴파일, 훈련

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=20, batch_size=1, verbose=1, validation_split=0.1)

loss, acc = model.evaluate(x_test, y_test)

print('loss : ', loss)
print('acc : ', acc)



