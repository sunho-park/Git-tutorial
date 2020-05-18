#1. 데이터

import numpy as np
x=np.array([range(1, 101), range(311, 411), range(100)])
y=np.array(range(711, 811))

x = np.transpose(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,shuffle=False, test_size=0.2)

# 2. 모델구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input
# model = Sequential()
# model.add(Dense(5, input_dim=3))
# model.add(Dense(5, input_shape(3,)))
# model.add(Dense(4))
# model.add(Dense(1))
input1 = Input(shape=(3, ))
dense1 = Dense(5, activation='relu')(input1)
dense1 = Dense(4)(dense1)
dense1 = Dense(4)(dense1)
dense1 = Dense(4)(dense1)
dense1 = Dense(4)(dense1)
dense1 = Dense(4)(dense1)
dense1 = Dense(4)(dense1)
dense1 = Dense(4)(dense1)
output1 = Dense(1)(dense1)

model = Model(inputs = input1, outputs=output1)

model.summary()

# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=300, batch_size=1, validation_split=0.25,verbose=3) 

# 4. 평가, 예측
loss, mse = model.evaluate(x_test, y_test, batch_size=1) 
print("loss : ", loss)
print("mse = ", mse)

y_predict = model.predict(x_test)
print("y_predict : \n", y_predict)

#RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict): 
    return np.sqrt(mean_squared_error(y_test, y_predict))
      
print("RMSE : ", RMSE(y_test, y_predict))

# R2 구하기

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("r2 : ", r2) 
