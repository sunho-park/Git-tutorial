#keras26_mlp1_hamsu
# 함수형 모델로 만드시오.
import numpy as np
from numpy import array
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input

# 1. 데이터

x = array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6],
            [5, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 10],
            [9, 10, 11], [10, 11, 12],
            [20, 30, 40],[30, 40, 50], [40, 50, 60]])       #(13, 3)

y = array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 50, 60, 70])       #(13, )


x_predict = array([50, 60, 70])  

x = x.reshape(13, 3, 1) 
x_predict = x_predict.reshape(1, 3, 1)

print("x.shape : ", x.shape)   #(13, 3)
print("y.shape : ", y.shape)   #(13, )

print("x : \n", x)
# 모델구성

#model.add(LSTM(10, activation='relu', input_shape=(3, 1))) # (none, 3, 1)

input1 = Input(shape=(3,1))
dense1_1 = LSTM(100)(input1)
dense1_1 = Dense(100)(dense1_1)
dense1_1 = Dense(100)(dense1_1)
dense1_1 = Dense(100)(dense1_1)
dense1_1 = Dense(100)(dense1_1)
dense1_1 = Dense(100)(dense1_1)
dense1_1 = Dense(7)(dense1_1)
dense1_1 = Dense(3)(dense1_1)

output1 = Dense(30)(dense1_1)
output1_1 = Dense(100)(output1)
output1_1 = Dense(100)(output1_1)
output1_1 = Dense(100)(output1_1)
output1_1 = Dense(100)(output1_1)
output1_1 = Dense(100)(output1_1)
output1_1 = Dense(100)(output1_1)
output1_1 = Dense(100)(output1_1)
output1_1 = Dense(1)(output1_1)

model = Model(inputs=input1, outputs=output1_1)
model.summary()



# 3. 실행
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=2000)


print("x_predict.shape : ", x_predict.shape)
print("x_predict : \n", x_predict)
print("---------------------------------------------------------------")


y_predict = model.predict(x_predict)

print("y_predict : ", y_predict)




