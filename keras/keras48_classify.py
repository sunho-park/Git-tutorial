import numpy as np
from keras.layers import Dense, LSTM
from keras.models import Model, Sequential
# 1. 데이터 
x = np.array(range(1, 11))
y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])

print(x.shape) #(10,) 스칼라 10개, dimension=1
print(y.shape) #(10,)


# 2. 모델 구성
model=Sequential()
model.add(Dense(10, input_dim=1, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50))                                      # 이렇게 써도 activation 에 디폴트 값이 있다.
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='sigmoid'))                 # 1). 마지막 값에 sigmoid 를 곱해서 0 or 1 의 값을 출력한다. sigmoid 의 값 between 0 and 1

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc']) # 분류모델에서는 평가지표 acc 쓴다 0과 1로 출력값이 분명하므로, 출력값이 딱떨어지지않을때 mse
model.fit(x, y, epochs=100, batch_size=1)                                    # 2). 2진분류모델 binary_crossentropy 밖에없다. 
 
# 4. 평가, 예측
loss, acc = model.evaluate(x, y, batch_size=1)
print("loss : ", loss)
print("acc : ", acc)

x_predict = np.array([1, 2, 3])
y_predict = model.predict(x_predict)

print("y_predict : \n", y_predict)

print(type(y_predict))

for i in y_predict:
    if i >=0.5:
        print(1)
    else:
        print(0)

# y_predict 값이 0,1 로 나오게 작성
#
# 1, sigmoid 에 적용할 함수 수식을 만든다.
# 2, 남이잘만든것이 있는지 찾는다.
# 3, 다른 방법한가지가 더 있음