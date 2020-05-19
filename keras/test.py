#1. 데이터

import numpy as np

x=np.array([range(1, 101), range(301, 401)])

y=np.array([range(711, 811), range(611, 711)])
y2=np.array([range(101, 201), range(411, 511)])

x = np.transpose(x)

y = np.transpose(y)
y2 = np.transpose(y2)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test, y2_train, y2_test = train_test_split(x, y, y2, shuffle=False, test_size=0.1)

# 모델구성

from keras.layers import Dense, Input
from keras.models import Model

input1 = Input(shape=(2, ))
dense = Dense(30)(input1)
dense = Dense(30)(dense)
dense = Dense(30)(dense)

from keras.layers.merge import concatenate
merge1 = Dense(10)(dense)
middle1 = Dense(10)(merge1)
middle1 = Dense(10)(merge1)

output1 = Dense(10)(middle1)
output1 = Dense(10)(output1)
output1 = Dense(2)(output1)

output2 = Dense(10)(middle1)
output2 = Dense(10)(output2)
output2 = Dense(2)(output2)

model = Model(inputs=input1, outputs=[output1, output2])

model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, [y_train, y2_train], epochs=100, batch_size=1, validation_split=0.5, verbose=2)

loss = model.evaluate(x_test, [y_test, y2_test], batch_size=1)
print("loss : ", loss)
y1_predict, y2_predict = model.predict(x_test)

print("y1_predict : ", y1_predict)
print("y2_predict : ", y2_predict)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
RMSE1 = RMSE(y_test, y1_predict)
RMSE2 = RMSE(y2_test, y2_predict)
print("RMSE1 : ", RMSE1)
print("RMSE2 : ", RMSE2)
print("RMSE : ", (RMSE1+RMSE2)/2)

from sklearn.metrics import r2_score
r2_1=r2_score(y_test, y1_predict)
r2_2=r2_score(y2_test, y2_predict)

print("r2_1 : ", r2_1)
print("r2_2 : ", r2_2)
print("r2 : ", (r2_1+r2_1)/2)







