from sklearn.datasets import load_boston


# (x_train, y_train), (x_test, y_test) = cifar100.load_data()
'''
data    : x 값
target  : y 값
'''
dataset = load_boston()
x  = dataset.data
y = dataset.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=False, test_size=0.1)

# print('x_train[0] : ', x_train[0])
# print('y_train[0] : \n', y_train[0])

print("x_train : \n", x_train)                        # (455, 13)
print("x_test : ", x_test)                            # (51, 13)
print("x_train.shape : ", x_train.shape)
print("x_test.shape : ", x_test.shape)

print("y_train : \n", y_train)
print("y_test : ", y_test)
print("y_train.shape : ", y_train.shape)          # (455, )
print("y_test.shape : ", y_test.shape)            # (51, )


# 정규화?
from sklearn.preprocessing import MinMaxScaler, StandardScaler, robust_scale, normalize

scaler = MinMaxScaler()        #정규화
# scaler = StandardScaler()       #표준화
# scaler = robust_scale()

scaler.fit(x)
x_train = scaler.transform(x_train)  #fit 하고 tranform 해야함.
x_test = scaler.transform(x_test)

print("x_train : \n", x_train)
print("x_test : \n", x_test)
print("x_test : ", x_test)


x_train = x_train.reshape(455, 13, 1)
x_test = x_test.reshape(51, 13, 1)

print("x_train.shape : ", x_train.shape)          
print("x_test.shape : ", x_test.shape) 

# 모델구성

from keras.models import Sequential

from keras.layers import Dense, LSTM


model = Sequential()

model.add(LSTM(10, input_shape = (13, 1), return_sequences=False))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

model.summary()

# 컴파일 훈련

model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=5, batch_size=1, verbose=1)

# 훈련 평가

loss, mse = model.evaluate(x, y, batch_size=1)

print("loss : ", loss)
print("mse : ", mse)


output = model.predict(x_test)

print("output : \n", output)



