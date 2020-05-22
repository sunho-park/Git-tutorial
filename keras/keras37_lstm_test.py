from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

#실습 : LSTM 레이어를 5개 이상 엮어서 Dense 결과를 이겨내시오!!! 85점 이상

# 1. 데이터

x = array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6],
            [5, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 10],
            [9, 10, 11], [10, 11, 12],
            [200, 300, 400],[300, 400, 500], [400, 500, 600]])            #(13, 3)
 
y = array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 50, 60, 70])         #(13, )
x_predict = array([55, 65, 75])                                   #(3, )

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
model.add(LSTM(100, activation='relu', input_shape=(3, 1), return_sequences=True)) # (none, 3, 1) 10은 노드의 갯수
model.add(LSTM(100, activation='relu', return_sequences=True))
model.add(LSTM(100, activation='relu', return_sequences=True))
model.add(LSTM(100, activation='relu', return_sequences=True))
# model.add(LSTM(10, input_length=3, input_dim=1, return_sequences=True))   return_sequence = True 는 차원을 유지시켜준다. 3차원
model.add(LSTM(100, return_sequences=False)) # LSTM 의 출력 값은 2차원
model.add(Dense(100))  #2차원
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))  
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(70))  #2차원
model.add(Dense(1))
model.summary()



# 3. 실행
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=500)

# print("x_predict", x_predict)
# print("x_predict.shape", x_predict.shape)
x_predict = x_predict.reshape(1, 3, 1)

print("x_predict : \n",x_predict)
print("x_predict.shape", x_predict.shape)

y_predict = model.predict(x_predict)
print("y_predict : ", y_predict)


