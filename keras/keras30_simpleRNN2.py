from numpy import array
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN

# 1. 데이터

x = array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6],
            [5, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 10],
            [9, 10, 11], [10, 11, 12],
            [20, 30, 40],[30, 40, 50], [40, 50, 60]])       #(13, 3)

y = array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 50, 60, 70])       #(13, )

print("x.shape : ", x.shape)   #(13, 3)
print("y.shape : ", y.shape)   #(13, )

print("x : \n", x)

# x = x.reshape(4, 3, 1) 
x = x.reshape(x.shape[0], x.shape[1], 1)

print("x : \n ", x)
print("x.shape", x.shape)

# 모델구성
model = Sequential()
#model.add(LSTM(10, activation='relu', input_shape=(3, 1))) # (none, 3, 1)
model.add(SimpleRNN(10, input_length=3, input_dim=1))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))
model.summary()


# 3. 실행
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100)

x_predict = array([50, 60, 70])  

print("x_predict", x_predict)
print("x_predict.shape", x_predict.shape)

print("---------------------------------------------------------------")

x_predict = x_predict.reshape(1, 3, 1)

print("x_predict : \n", x_predict)
print("x_predict : ", x_predict.shape)
y_predict = model.predict(x_predict)
print("y_predict : ", y_predict)




