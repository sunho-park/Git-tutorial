from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

# 1. 데이터

x = array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]])

y = array([4, 5, 6, 7])        #(4, )마지막 괄호는 계산 하지 않기    


'''
y2= array([[4, 5, 6, 7]])      #(1, 4)
y3 = array([[4],[5],[6],[7]])  #(4, 1)'''
# import numpy as np
# X = np.array

print("x.shape : ", x.shape)   
print("y.shape : ", y.shape)   

print("x : \n", x)
# x = x.reshape(4, 3, 1) 
x = x.reshape(x.shape[0], x.shape[1], 1)

'''
                행,         열,     몇개씩 자르는지.
x의 shape = (batch_size, timesteps, feature)
iuput_shape = (timesteps, feature)
input_length = timesteps, 
input_dim = feature

lstm 3차원 인풋
dense 2차원

'''

print("x : \n ", x)
print("x.shape", x.shape)

# 모델구성
model = Sequential()
#model.add(LSTM(10, activation='relu', input_shape=(3, 1))) # (none, 3, 1)
model.add(LSTM(7, input_length=3, input_dim=1))
model.add(Dense(4))
model.add(Dense(1))
model.summary()


# 3. 실행
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=10)

x_predict = array([5, 6, 7])  # 스칼라

print("x_predict", x_predict)
print("x_predict.shape", x_predict.shape)
x_predict = x_predict.reshape(1, 3, 1)

print("x_predict : ",x_predict)
print("x_predict.shape", x_predict.shape)

y_predict = model.predict(x_predict)
print(y_predict)

#문제점 1. 데이터가 너무 적다. 2.





