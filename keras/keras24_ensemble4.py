#1. 데이터

import numpy as np

x1=np.array([range(1, 101), range(301, 401)])

y1=np.array([range(711, 811), range(611, 711)])
y2=np.array([range(101, 201), range(411, 511)])

x1 = np.transpose(x1)

y1 = np.transpose(y1)
y2 = np.transpose(y2)

from sklearn.model_selection import train_test_split

# x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, shuffle=False, test_size=0.1)
# y2_train, y2_test = train_test_split(y2, shuffle=False, test_size=0.1)

x1_train, x1_test, y1_train, y1_test, y2_train, y2_test = train_test_split(x1, y1, y2, shuffle=False, test_size=0.1)

print(x1_train.shape) #(90.2)
print(x1_test.shape)  #(10,2)
print(y1_train.shape)
print(y2_train.shape)

# 2. 모델구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input

input1 = Input(shape=(2, ))
dense1_1 = Dense(5, activation='relu', name='bit1')(input1)
dense1_2 = Dense(4, activation='relu', name='bit2')(dense1_1)
dense1_2 = Dense(4, activation='relu', name='bit3')(dense1_2)

# from keras.layers.merge import concatenate
# merge1 = concatenate([dense1_2, dense2_2])

# middle1 = Dense(15)(merge1)
# middle1 = Dense(5)(middle1)
# middle1 = Dense(7)(middle1)

######## output 모델 구성 #######

output1 = Dense(30)(dense1_2)
output1_1 = Dense(5)(output1)
output1_1 = Dense(2)(output1_1)

output2 = Dense(30)(dense1_2)
output2_1 = Dense(7)(output2)
output2_2 = Dense(2)(output2_1)

model = Model(inputs = [input1], outputs=[output1_1, output2_2])
model.summary()


# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x1_train, [y1_train, y2_train], epochs=10, batch_size=1, validation_split=0.25, verbose=1) 

# 4. 평가, 예측

loss = model.evaluate(x1_test, [y1_test, y2_test], batch_size=1)
# mse = model.evaluate([x1_test, x2_test], [y1_test, y2_test], batch_size=1)

print("loss : ", loss)
#print("mse = ", mse)

y1_predict, y2_predict = model.predict(x1_test) #(20,2)가 1개가 들어감 출력값 2개이므로

# y1_predict = model.predict(x1_test) (20,2)가 2개가 들어감
# y2_predict = model.predict(x1_test) (20,2)가 2개가 들어감

print("y1_predict : \n", y1_predict)
print("y2_predict : \n", y2_predict)

#RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict): 
    return np.sqrt(mean_squared_error(y_test, y_predict))

RMSE1 = RMSE(y1_test, y1_predict)
RMSE2 = RMSE(y2_test, y2_predict)

print("RMSE1 : ", RMSE1)
print("RMSE2 : ", RMSE2)
print("RMSE : ", (RMSE1+RMSE2)/2)

# R2 구하기

from sklearn.metrics import r2_score
r2_1 = r2_score(y1_test, y1_predict)
r2_2 = r2_score(y2_test, y2_predict)


print("r2_1 : ", r2_1)
print("r2_2 : ", r2_2)
print("R2 : ", (r2_1+r2_2)/2)
