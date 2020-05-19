
#1. 데이터

import numpy as np
x=np.array([range(1, 101), range(311, 411), range(100)])
y=np.array([range(101, 201), range(711, 811),range(100)])


print(x.shape) #(3,100)

#행과 열을 바꾸는 함수  (3,100)--->(100,3)
x = np.transpose(x)
y = np.transpose(y)

print(x.shape) #(100,3)
print(y.shape)

#x=np.swapaxes(x,0,1)
#x=np.arange(300).reshape(100,3)

# print("x = ", x)
# print("y = ", y)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)  #train (80,3) test(20,3)

#x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, random_state=66, test_size=0.66)

#x, y, random_state=99, shufle=True, test_size=0.4

#x_test, y_test, random_state=99, test_size=0.5

# x_train = x[:60] #0~59
# x_val = x[60:80] #60~79
# x_test = x[80:]  #80~66

# y_train = x[:60]
# y_val = x[60:80]
# y_test = x[80:]

# print("x_train = \n", x_train)
# print("y_train = \n", y_train)
# print("x_test = \n", x_test)
# print("y_test = \n", y_test)

# print("x_val = ", x_val)
# print("y_val = ", y_val)'''



# 2. 모델구성
from keras.models import Sequential
from keras.layers import Dense

# Sequential 함수를 model 로 하겠다.

model = Sequential()

model.add(Dense(5, input_dim=3)) #1~100의 한 덩어리?
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(3))

# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])



model.fit(x_train, y_train, epochs=10, batch_size=1, validation_split=0.25)       # x_train : (60,3) x_val :(20,3), x_test :(20,3)
 #훈련용 데이터로 훈련 ,
# validation_data = (x_test, y_test), 

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
print("r2 : ", r2) #회귀모형'''
