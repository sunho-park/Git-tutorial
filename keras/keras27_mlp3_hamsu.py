# keras15_mlp 를 Sequential 에서 함수형으로 변경
# earlystopping 적용
#1. 데이터

import numpy as np
x=np.array([range(1, 101), range(311, 411), range(100)])
y=np.array(range(711, 811))


#행과 열을 바꾸는 함수
x = np.transpose(x)


print("x.shape = ", x.shape)
print("y.shape = ", y.shape)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, test_size=0.2)

# 2. 모델구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input

input1=Input(shape=(3,))
dense1_1 = Dense(30)(input1)
dense1_1 = Dense(10)(dense1_1)
dense1_1 = Dense(5)(dense1_1)

output1 = Dense(30)(dense1_1)
output1_1 = Dense(10)(output1)
output1_2 = Dense(1)(output1_1)

model = Model(inputs = input1, outputs=output1_2)
model.summary()

# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'loss', patience=5, mode='min')

model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.25, verbose=1, callbacks=[early_stopping])       # train (60.3), val(20.3),test(20,3)

# 4. 평가, 예측
loss, mse = model.evaluate(x_test, y_test, batch_size=1) 
print("loss : ", loss)
print("mse = ", mse)


y_predict = model.predict(x_test)
print("y_predict : \n", y_predict)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict): 
    return np.sqrt(mean_squared_error(y_test, y_predict))
      
print("RMSE : ", RMSE(y_test, y_predict))

# R2 구하기

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("r2 : ", r2)