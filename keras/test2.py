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
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()      

scaler.fit(x)
x = scaler.transform(x) 


# print('x_train[0] : ', x_train[0])
# print('y_train[0] : \n', y_train[0])
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

model.compile(loss='mse', optimizer='adam', metrics=['acc'])
model.fit(x, y, epochs=20, batch_size=1, verbose=1, validation_split=0.1)

# 훈련 평가

loss, acc = model.evaluate(x, y)

print("loss : ", loss)
print("acc : ", acc)


# output = model.predict(x_test)

# print("output : \n", output)



