#1. 데이터
import numpy as np
x=np.array(range(1, 101))
y=np.array(range(101, 201))

x_train = x[:60] #0~59
x_val = x[60:80] #60~79
x_test = x[80:]  #80~99

y_train = x[:60]
y_val = x[60:80]
y_test = x[80:]

print("x_train = ", x_train)
print("y_train = ", x_train)
print("x_val = ", x_val)
print("y_val = ", y_val)
print("x_test = ", x_test)
print("y_test = ", y_test)

# 2. 모델구성
from keras.models import Sequential
from keras.layers import Dense

# Sequential 함수를 model 로 하겠다.

model = Sequential()

model.add(Dense(5, input_dim=1)) #1~100의 한 덩어리?

model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=10, batch_size=1, validation_data = (x_val, y_val)) #훈련용 데이터로 훈련

# 4. 평가, 예측
loss, mse = model.evaluate(x_test, y_test, batch_size=1) #훈련용 데이터와 평가용 데이터를 분리해야함
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
print("r2 : ", r2) #회귀모형

