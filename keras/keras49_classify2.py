import numpy as np
from keras.layers import Dense, LSTM
from keras.models import Model, Sequential



# 이중분류, 다중분류
# loss, output layer 변경   전에 y값이 변경되있어야함
# 데이터 전처리, 와꾸 맞춰줘야함

# 1. 데이터 
x = np.array(range(1, 11))
y = np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5]) #스칼라 10개 벡터 1 디멘션 1 

# 1번째 방법

print("y : \n", y)
print("y.shape : ", y.shape) #(10, ) -> (10, 1)

y = y.reshape(10, 1) # or y= y.reshape(-1, 1)

from sklearn.preprocessing import OneHotEncoder

aaa = OneHotEncoder()
aaa.fit(y)
y = aaa.transform(y).toarray()      

print("y : \n", y)
print("y.shape : ", y.shape)

print("======================================================")
'''
# 2번째 방법

y= y - 1

from keras.utils import np_utils
y = np_utils.to_categorical(y)

print("x.shape : ", x.shape)
print("y.shape : ", y.shape) # (10, 5)
print("y : \n", y)

print("======================================================")

# 3번째 방법

print(x.shape) 
print(y.shape)   # (10, 6)
print(y) 

from keras.utils import np_utils # one-hot 인코딩 1차원이 2차원이 된다.
y = np_utils.to_categorical(y)

y=y[:, 1:6]

print(y.shape)   # (10, 5)
print(y) 
'''
# 2. 모델 구성

model = Sequential()
model.add(Dense(10, input_dim=1, activation='relu'))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(5, activation='softmax')) # y.shape = (10, 6)
model.summary()

# 3. 컴파일, 훈련

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc']) #분류모델에서는 acc
model.fit(x, y, epochs=10, batch_size=1)


# 4. 예측, 평가

loss, acc = model.evaluate(x, y)

print("loss : ", loss)
print("acc : ", acc)

x_predict = np.array([1, 2, 3, 4, 5])
print(x_predict.shape)                  # (5, )
y_predict = model.predict(x_predict)
print("y_predict : \n", y_predict)
print("y_predict.shape : \n", y_predict.shape) # (5, 5)

y_predict = np.argmax(y_predict, axis=1).reshape(-1)
print("y_predict : \n", y_predict)

# axis 제거할 축을 입력한다.
# reshape ()의 -1이 의미하는 바는 변경된 배열의 -1 위치의 차원은 원래 배열의 길이와 남은 차원으로 부터 추정 rfriend.tistory.com/345
# 과제 dim 을 6에서 5로 변경
# predict도 구하기


