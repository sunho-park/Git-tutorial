from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM 
import numpy as np
# 1. 데이터

x = array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6],
            [5, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 10],
            [9, 10, 11], [10, 11, 12],
            [20, 30, 40],[30, 40, 50], [40, 50, 60]])            #(13, 3)
 
y = array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 50, 60, 70])         #(1,13 )
x_predict = array([55, 65, 75])                                  # (1, 3)

print("x.shape : ", x.shape)   
print("y.shape : ", y.shape)   


print("x : \n", x)
print("y : \n ", y)


# print("x.shape : ", x.shape)
# print("y.shape : ", y.shape)  
# x = x.reshape(x.shape[0], x.shape[1], 1)

'''
                행,         열,     몇개씩 자르는지.
x의 shape = (batch_size, timesteps, feature)
iuput_shape = (timesteps, feature)
input_length = timesteps, 
input_dim = feature

lstm 3차원 인풋
dense 2차원
'''

# 모델구성
model = Sequential()
model.add(Dense(13, input_dim=3))
model.add(Dense(13)) 
model.add(Dense(13)) 
model.add(Dense(13)) 
model.add(Dense(13)) 
model.add(Dense(13)) 
model.add(Dense(13)) 
model.add(Dense(13)) 
model.add(Dense(5)) 
model.add(Dense(1))

model.summary()



# 3. 실행
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=500)


print("x_predict", x_predict)
print("x_predict.shape", x_predict.shape)

# print("y_predict : ",y_predict)
# print("y_predict.shape", y_predict.shape)

x_predict = x_predict.reshape(1, 3)

print("x_predict", x_predict)
print("x_predict.shape", x_predict.shape)

y_predict = model.predict(x_predict)

print("y_predict.shape : ", y_predict.shape)
print("y_predict : ", y_predict)



