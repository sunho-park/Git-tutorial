from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

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
input_length = timesteps
input_dim = feature

lstm 3차원 인풋
dense 2차원
'''
print("x : \n ", x)
print("x.shape", x.shape)

# 모델구성
model = Sequential()
model.add(LSTM(10, activation='relu', input_shape=(3, 1), return_sequences=True)) # (none, 3, 10) 10은 노드의 갯수
# model.add(LSTM(10, input_length=3, input_dim=1, return_sequences=True))   return_sequence = True 는 차원을 유지시켜준다. 3차원
model.add(LSTM(10, return_sequences=False)) # LSTM 의 출력 값은 2차원
model.add(Dense(5) )  #2차원
model.add(Dense(1))
model.summary()

''' 
lstm_1(LSTM) shape 10 이 나온이유 : LSTM 10 은 노드의 갯수는 그 다음 차원 시작의 feature다. 
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
lstm_1 (LSTM)                (None, 3, 10)             480
_________________________________________________________________
lstm_2 (LSTM)                (None, 10)                840
_________________________________________________________________
dense_1 (Dense)              (None, 5)                 55
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 6
=================================================================
Total params: 1,381
Trainable params: 1,381
Non-trainable params: 0 '''


# 3. 실행
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100)

x_predict = array([5, 6, 7])  # 스칼라

print("x_predict : \n", x_predict)
print("x_predict.shape : ", x_predict.shape)

x_predict = x_predict.reshape(1, 3, 1)

print("x_predict :\n",x_predict)
print("x_predict.shape :", x_predict.shape)

y_predict = model.predict(x_predict)
print("y_predict : ", y_predict)



