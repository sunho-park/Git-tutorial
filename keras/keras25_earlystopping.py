#1. 데이터

import numpy as np

x1=np.array([range(1, 101), range(311, 411), range(411, 511)])
x2=np.array([range(711, 811), range(711, 811), range(511, 611)])

y=np.array([range(101, 201), range(411, 511), range(100)])

x1 = np.transpose(x1)
x2 = np.transpose(x2)

y = np.transpose(y)

from sklearn.model_selection import train_test_split

x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, y, x2, shuffle=False, test_size=0.2) 


# 2. 모델구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input

input1 = Input(shape=(3, ))
dense1_1 = Dense(5, activation='relu', name='bit1')(input1)
dense1_2 = Dense(4, activation='relu', name='bit2')(dense1_1)
dense1_2 = Dense(4, activation='relu', name='bit3')(dense1_2)

input2 = Input(shape=(3, ))
dense2_1 = Dense(10, activation='relu')(input2)
dense2_2 = Dense(5, activation='relu')(dense2_1)

from keras.layers.merge import concatenate
merge1 = concatenate([dense1_2, dense2_2])

middle1 = Dense(15)(merge1)
middle1 = Dense(5)(middle1)
middle1 = Dense(7)(middle1)

######## output 모델 구성 #######
output1 = Dense(30)(middle1)
output1_1 = Dense(7)(output1)
output1_1 = Dense(3)(output1_1)

model = Model(inputs = [input1, input2], outputs=output1_1)

model.summary()


# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'loss', patience=155, mode='min')
#그 시점에 돌입했을 때는 이미 안좋은 시점까지 들어간 구간. 따라서 가내수공업으로 그전값을 구해야함 팅기는 시점이 5번 정도 되는 시점의 mode의 종류 min, max, auto
model.fit([x1_train, x2_train], y_train, epochs=1000, batch_size=1, validation_split=0.25, verbose=1, callbacks=[early_stopping]) #기본적으로 리스트로 들어가있음

# 4. 평가, 예측

loss = model.evaluate([x1_test, x2_test], y_test, batch_size=1)

print("loss : ", loss)
#print("mse = ", mse)

y_predict= model.predict([x1_test, x2_test])  #(20, 3)

print("y_predict : \n", y_predict)

#RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict): 
    return np.sqrt(mean_squared_error(y_test, y_predict))

print("RMSE : ", RMSE(y_test, y_predict))

# R2 구하기

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)
