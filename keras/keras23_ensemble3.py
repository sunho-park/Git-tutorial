#1. 데이터

import numpy as np

x1=np.array([range(1, 101), range(311, 411), range(411, 511)])
x2=np.array([range(711, 811), range(711, 811), range(511, 611)])

y=np.array([range(101, 201), range(411, 511), range(100)])

####여기서 부터 수정하세요#####
#############################
x1 = np.transpose(x1)
x2 = np.transpose(x2)
y = np.transpose(y)

from sklearn.model_selection import train_test_split
x1_train, x1_test, y_train, y_test,  x2_train, x2_test  = train_test_split(x1, y, x2, shuffle=False, test_size=0.2)
# x1_val, x1_test, y1_val, y1_test = train_test_split(x1_test, y1_test, test_size=0.5)
# x2_train, x2_test = train_test_split(x2,shuffle=False, test_size=0.2)
# x2_val, x2_test, y2_val, y2_test = train_test_split(x2_test, y2_test, test_size=0.5)


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
model.fit([x1_train, x2_train], y_train, epochs=100, batch_size=1, validation_split=0.5, verbose=1) 

# 4. 평가, 예측

loss = model.evaluate([x1_test, x2_test], y_test, batch_size=1)
# mse = model.evaluate([x1_test, x2_test], [y1_test, y2_test], batch_size=1)

print("loss : ", loss)
#print("mse = ", mse)

y1_predict, y2_predict = model.predict([x1_test, x2_test])  #(20, 3)


print("y1_predict : \n", y1_predict)
print("y2_predict : \n", y2_predict)


#RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict): 
    return np.sqrt(mean_squared_error(y_test, y_predict))

print("RMSE : ", RMSE(y_test, y_predict))

# R2 구하기

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("==============================================")
print("R2 : ", r2)
