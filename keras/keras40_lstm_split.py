
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

#1. 데이터
a = np.array(range(1, 11)) #1~10

size=5                       # time_steps=4 

#LSTM 모델을 완성하시오.

def split_x(seq, size): # seq 전체 데이터, size 크기
    aaa= []

    for i in range(len(seq) - size + 1):    #range(6)    # 0, 1, 2, 3, 4, 5
        print("i : ", i)
        subset = seq[i : (i+size)]
        print("subset : ", subset)
        aaa.append([item for item in subset])
        
   # print(type(aaa)) list
    return np.array(aaa)

dataset =split_x(a, size)
print("dataset : \n", dataset)
print("dataset.shape : \n", dataset.shape)
print("type(dataset) : \n", type(dataset))


         # 행, 열   tip: 함수의 목적은 재사용 그러므로 함수를 변경할떄 고려해야함
x =dataset[ :, :4] # or [:, 0:4],  [:] -> all
y=dataset[:, 4:5]  # or [:, 4]
print("x : \n", x)
print("y : \n", y)

x = x.reshape(6, 4, 1) # or x = x.reshape(x, (6, 4, 1))

print("x : \n", x)
print("x.shape", x.shape)

model = Sequential()
model.add(LSTM(10, activation='relu', input_shape=(4, 1)))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1)) # y.shape = (6, 1)


model.compile(loss='mse', optimizer='adam', metrics=['mse'])

from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor = 'loss', patience=50, mode='auto')
model.fit(x, y, epochs=1000, batch_size=1, verbose=1, callbacks=[early_stopping])

loss, mse = model.evaluate(x, y)

print("lose : ", loss)
print("mse : ", mse)


x_predict = np.array(range(15, 19)) # 15, 16, 17, 18

print("x_predict", x_predict)
print("x_predict.shape", x_predict.shape)
x_predict = x_predict.reshape(1, 4, 1)

print("x_predict : \n",x_predict)
print("x_predict.shape", x_predict.shape)

y_predict = model.predict(x_predict)

print("y_predict : ", y_predict)

