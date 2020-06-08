import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense

hite = np.load('./data/csv/하이트 주가.npy', allow_pickle=True)
samsung = np.load('./data/csv/삼성전자 주가.npy', allow_pickle=True)


# print('hite : ', hite)
# print('samsung : ', samsung)

print('hite.shape : ', hite.shape)           # (509, 5)
print('samsung.shape : ', samsung.shape)     # (509, 1)

def split_xy1(data, ts):
    a, b = list(), list()
    for i in range(len(data)):
        e_n= i +ts
        if e_n > len(data) - 1:
            break
        t_x, t_y = data[i:e_n], data[e_n]
        a.append(t_x)
        b.append(t_y)
    return np.array(a), np.array(b)

x1, y1 = split_xy1(samsung, 5)
'''
print('x1 : ', x1)
print('y1 : ', y1)
print('x1.shape : ', x1.shape)  # 504, 5, 1
print('y1.shape : ', y1.shape)  # 504, 1
'''

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

x2, y2 = split_xy5(hite, 5, 1)

print('x2 : ', x2) 
print('y2 : ', y2)
print('x1.shape : ', x1.shape)   # (504, 5, 1)
print('x2.shape : ', x2.shape)   # (504, 5, 5)
print('y2.shape : ', y2.shape)   # (504, 1)  

x1 = x1.reshape(504, 5)
x2 = x2.reshape(504, 25)
# 데이터 전처리

from sklearn.preprocessing import StandardScaler

scaler1 = StandardScaler()
scaler1.fit(x1)
x1 = scaler1.transform(x1)


scaler2 = StandardScaler()
scaler2.fit(x2)
x2 = scaler2.transform(x2)


# 데이터 셋 나누기

from sklearn.model_selection import train_test_split

x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, random_state=1, test_size=0.3)
x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, random_state=1, test_size=0.3)

print(x2_train.shape)       # 352, 5, 5
print(x2_test.shape)        # 152, 5, 5
print(y2_train.shape)       # 352, 1
print(y2_test.shape)        # 152, 1
'''
# reshape

x1_train = np.reshape(x1_train, (x1_train.shape[0], x1_train.shape[1] * x1_train.shape[2]))
x1_test = np.reshape(x1_test, (x1_test.shape[0], x1_test.shape[1]*x1_test.shape[2]))
x2_train = np.reshape(x2_train, (x2_train.shape[0], x2_train.shape[1]*x2_train.shape[2]))
x2_test = np.reshape(x2_test, (x2_test.shape[0], x2_test.shape[1]*x2_test.shape[2]))

print('x1_train : ', x1_train.shape)         # 352, 5
print('x1_test.shape : ', x1_test.shape)     #  152, 5

print('x2_train : ', x2_train.shape) # (352, 25)
print('x2_test.shape : ', x2_test.shape)  # (152, 25)


# 데이터 전처리

from sklearn.preprocessing import StandardScaler

scaler1 = StandardScaler()
scaler1.fit(x1_train)

x1_train_scaled = scaler1.transform(x1_train)
x1_test_scaled = scaler1.transform(x1_test)

scaler2 = StandardScaler()
scaler2.fit(x2_train)

x2_train_scaled = scaler2.transform(x2_train)
x2_test_scaled = scaler2.transform(x2_test)

print(x2_train_scaled[0, :])
'''

# 모델 구성

from keras.models import Model
from keras.layers import Dense, Input

input1 = Input(shape=(5, ))
dense1 = Dense(64)(input1)
dense1 = Dense(32)(dense1)
dense1 = Dense(32)(dense1)
dense1 = Dense(32)(dense1)
output1 = Dense(32)(dense1)

input2 = Input(shape=(25, ))
dense2 = Dense(64)(input2)
dense2 = Dense(32)(dense2)
dense2 = Dense(32)(dense2)
dense2 = Dense(32)(dense2)
dense2 = Dense(32)(dense2)
output2 = Dense(32)(dense2)

from keras.layers.merge import concatenate
merge = concatenate([output1, output2])
output3 = Dense(1)(merge)

model = Model(inputs = [input1, input2], outputs = output3)

# 컴파일 훈련

model.compile(loss='mse', optimizer='adam', metrics=['mse'])

from keras.callbacks import EarlyStopping
es = EarlyStopping(patience=20)
model.fit([x1_train, x2_train], y1_train, validation_split=0.2, verbose=1 , batch_size=1, epochs=10, callbacks=[es]) 


# 평가 / 예측
loss, mse = model.evaluate([x1_test, x2_test], y1_test, batch_size=1)

print("loss : ", loss)
print("mse : ", mse)

print('x1_test :', x1_test)
print('x2_test : ', x2_test)

print('x1_test.shape : ', x1_test.shape)    # (152, 5)
print('x2_test.shape : ', x2_test.shape)    # (152, 25)      


y1_pred = model.predict([x1_test, x2_test])

print('y1_pred : ',y1_pred)

for i in range(5):
    print('종가 : ', y1_test[i], '/ 예측가 : ', y1_pred[i])

print('x1.shape : ', x1.shape)     # (504, 5)
print('x2.shape : ', x2.shape)     # (504, 25)

print('x1[-1]', x1[-1])
print('x2[-1]', x2[-1])

print('x1[-1].shape', x1[-1].shape)     # (5, )
print('x2[-1].shape', x2[-1].shape)     # (25, )

y1_predict = model.predict([[x1[-1]], [x2[-1]]])
print('y1_predict : ', y1_predict)


#y1_predict :  [[51268.457]]
