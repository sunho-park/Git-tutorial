from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

# 1. 데이터

x = array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6 ,7]])

y = array([6, 7, 8])        

print("x.shape : ", x.shape)   #(3, 5)
print("y.shape : ", y.shape)   #(3, ) 스칼라3개짜리

print(x)
x = x.reshape(x.shape[0], x.shape[1], 1)
print('--------------------------------------------')
print(x)
print(x.shape)    #(3, 5, 1) (3행, 5열, 1feature)  x의 컬럼(열)과 y의 벡터값을 일치  X의 행의수 = y 벡터 크기일치

# 모델구성
model = Sequential()
model.add(LSTM(7, activation='relu', input_shape=(5, 1))) # (None, 3, 1) input_shape를 행과 열로 몇개씩 자르는지 shape 에 대해 설명하지만 정식 용어로는 samples, time steps, feature
model.add(Dense(4))                                       # (행, 열, feature)
model.add(Dense(1))
model.summary()

# 3. 실행
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1)

x_predict = array([[4, 5, 6, 7, 8]])  # 스칼라

print("x_predict", x_predict)                   
print("x_predict.shape", x_predict.shape)           # (1,5)
                                          
x_predict = x_predict.reshape(x_predict.shape[0], x_predict.shape[1], 1)      

print("x_predict.shape : ", x_predict.shape)        #(1, 5, 1)
print("x_predict : ", x_predict)

y_predict = model.predict(x_predict)
print("y.predict : ", y_predict)






