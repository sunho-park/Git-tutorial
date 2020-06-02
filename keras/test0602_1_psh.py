import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense

hite = np.load('./data/csv/하이트 주가.npy', allow_pickle=True)
samsung = np.load('./data/csv/삼성전자 주가.npy', allow_pickle=True)

print('hite : ', hite)
print('samsung : ', samsung)

print('hite.shape : ', hite.shape)           # (508, 5)
print('samsung.shape : ', samsung.shape)     # (509, 1)
 
# hite 결측치 보완

'''
def split_hite3(dataset, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column

        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i:x_end_number, :]
        tmp_y = dataset[x_end_number : y_end_number, :]
        
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)
x, y = split_hite3(hite, 3, 1)


print('x : ', x)
print('y : ', y)
print("x.shape : ", x.shape)     
print("y.shape : ", y.shape)

# 데이터 셋 나누기

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, test_size=0.3)

print('x_train.shape : ', x_train.shape)
print('x_test.shape : ', x_test.shape)
print('y_train.shape : ', y_train.shape)
print('y_test.shape : ', y_test.shape)

print('x_test : ', x_test)

# 표준화를 위한 reshape

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1]*x_train.shape[2]))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1]*x_test.shape[2]))

y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1]*y_train.shape[2]))
y_test = np.reshape(y_test, (y_test.shape[0], y_test.shape[1]*y_test.shape[2]))


print('x_train.shape : ', x_train.shape) # (353, 15)
print('x_test.shape : ', x_test.shape)   # (152, 15)

print('y_train.shape : ', y_train.shape) # (353, 5)
print('y_test.shape : ', y_test.shape)   # (152, 5)

# 표준화


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

print(x_train_scaled[0, :])


# 모델 구성

model = Sequential()
model.add(Dense(64, input_shape=(15, )))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(5))

# 훈련

model.compile(loss='mse', optimizer='adam', metrics=['mse'])
from keras.callbacks import EarlyStopping
es = EarlyStopping(patience=20)

model.fit(x_train, y_train, validation_split=0.2, verbose=1, batch_size=1, epochs=100, callbacks=[es])

#평가

loss, mse = model.evaluate(x_test, y_test, batch_size=1)

print('loss : ', loss)
print('mse : ', mse)
 
print(type(x_test))
x_predict = np.array([[36000, 38750, 36000, 38750, 1407345, 
                     35900, 36750, 35900, 36000, 576566, 
                     36200, 36300, 35500, 35800, 548493]])

y_predict = model.predict(x_predict)

print("y_predict : \n", y_predict)
'''



