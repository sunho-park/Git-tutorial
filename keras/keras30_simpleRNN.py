<<<<<<< HEAD
from numpy import array
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN

# 1. 데이터

x = array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]])

y = array([4, 5, 6, 7])

print("x.shape : ", x.shape)   
print("y.shape : ", y.shape)   

print("x : \n", x)
# x = x.reshape(4, 3, 1) 
x = x.reshape(x.shape[0], x.shape[1], 1)


print("x : \n ", x)
print("x.shape", x.shape)

# 모델구성
model = Sequential()
#model.add(LSTM(10, activation='relu', input_shape=(3, 1))) # (none, 3, 1)
model.add(SimpleRNN(7, input_length=3, input_dim=1))
model.add(Dense(4))
model.add(Dense(1))
model.summary()


# 3. 실행
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100)

x_predict = array([5, 6, 7]) 

print("x_predict : ", x_predict)
print("x_predict.shape : ", x_predict.shape)
print("---------------------------------------------------------------")

x_predict = x_predict.reshape(1, 3, 1)

print("x_predict : \n",x_predict)
print("x_predict.shape", x_predict.shape)

y_predict = model.predict(x_predict)
print(y_predict)






=======
from numpy import array
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN

# 1. 데이터

x = array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]])

y = array([4, 5, 6, 7])

print("x.shape : ", x.shape)   
print("y.shape : ", y.shape)   

print("x : \n", x)
# x = x.reshape(4, 3, 1) 
x = x.reshape(x.shape[0], x.shape[1], 1)


print("x : \n ", x)
print("x.shape", x.shape)

# 모델구성
model = Sequential()
#model.add(LSTM(10, activation='relu', input_shape=(3, 1))) # (none, 3, 1)
model.add(SimpleRNN(7, input_length=3, input_dim=1))
model.add(Dense(4))
model.add(Dense(1))
model.summary()


# 3. 실행
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100)

x_predict = array([5, 6, 7]) 

print("x_predict : ", x_predict)
print("x_predict.shape : ", x_predict.shape)
print("---------------------------------------------------------------")

x_predict = x_predict.reshape(1, 3, 1)

print("x_predict : \n",x_predict)
print("x_predict.shape", x_predict.shape)

y_predict = model.predict(x_predict)
print(y_predict)






>>>>>>> eb8379fdc37cda08b49b7bec4a0c90005b484593
