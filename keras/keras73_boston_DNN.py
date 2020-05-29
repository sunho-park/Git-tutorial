from sklearn.datasets import load_boston


# (x_train, y_train), (x_test, y_test) = cifar100.load_data()
'''
data    : x 값
target  : y 값
'''
dataset = load_boston()
x  = dataset.data
y = dataset.target

print("x : ", x)

# 정규화
from sklearn.preprocessing import MinMaxScaler, StandardScaler, robust_scale, normalize

scaler = MinMaxScaler()      

scaler.fit(x)
x = scaler.transform(x) 

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=False, test_size=0.1)

# print('x_train[0] : ', x_train[0])
# print('y_train[0] : \n', y_train[0])

print("x_train : \n", x_train)                        # (455, 13)
print("x_train[0] : \n", x_train[0])
print("x_test : ", x_test)                            # (51, 13)
print("x_train.shape : ", x_train.shape)
print("x_test.shape : ", x_test.shape)

print("y_train : \n", y_train)
print("y_test : ", y_test)
print("y_train.shape : ", y_train.shape)          # (455, )
print("y_test.shape : ", y_test.shape)            # (51, )




# print("x_train : \n", x_train)
# print("x_test : \n", x_test)


# 모델구성

from keras.models import Sequential

from keras.layers import Dense


model = Sequential()

model.add(Dense(10, input_shape = (13, )))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(1))

model.summary()

# 컴파일 훈련

model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=20, batch_size=1, verbose=1, validation_split=0.1)

# 훈련 평가

loss, mse = model.evaluate(x_test, y_test)

print("loss : ", loss)
print("mse : ", mse)

