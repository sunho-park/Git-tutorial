from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM


#실습 : LSTM 레이어를 5개 이상 엮어서 Dense 결과를 이겨내시오!!! 85점 이상


# 1. 데이터

x = array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]])

y = array([4, 5, 6, 7])        #(4, )마지막 괄호는 계산 하지 않기    


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
model.add(Dense(10)) 
model.add(Dense(10)) 
model.add(Dense(5))  #2차원
model.add(Dense(1))
model.summary()



# 3. 실행
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100)

x_predict = array([5, 6, 7])  # 스칼라

print("x_predict", x_predict)
print("x_predict.shape", x_predict.shape)
x_predict = x_predict.reshape(1, 3, 1)

print("x_predict : ",x_predict)
print("x_predict.shape", x_predict.shape)

y_predict = model.predict(x_predict)
print("y_predict : ", y_predict)


