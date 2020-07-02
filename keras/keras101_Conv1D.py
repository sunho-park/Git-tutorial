
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv1D, MaxPool1D, Flatten

#1. 데이터
a = np.array(range(1, 101))  
size=5                       # time_steps=4 

#LSTM 모델을 완성하시오.

def split_x(seq, size): # seq 전체 데이터, size 크기
    aaa= []

    for i in range(len(seq) - size + 1):    # x 컬럼 96행
        #print("i : ", i)
        subset = seq[i : (i+size)]
        #print("subset : ", subset)
        aaa.append([item for item in subset])
        
   # print(type(aaa)) list
    return np.array(aaa)

dataset =split_x(a, size)
print("dataset : \n", dataset)
print("dataset.shape : \n", dataset.shape)  #(96, 5)
print("type(dataset) : \n", type(dataset))


         # 행, 열   tip: 함수의 목적은 재사용 그러므로 함수를 변경할떄 고려해야함
x=dataset[:90, :4] # or [:, 0:4],  [:] -> all
y=dataset[:90, 4:5]  # or [:, 4]
print("x : \n", x)
print("y : \n", y)
print("x.shape", x.shape)
print("y.shape", y.shape)

x = x.reshape(90, 4, 1) # or x = x.reshape(x, (90, 4, 1))

print("x : \n", x)
print("x.shape", x.shape)

# 실습 1. train , test 분리할것. (8:2)
# 실습 2. 마지막 6개의 행을 predict 로 만들고 싶다.
# 실습 3. validation을 넣을 것 (train의 20%)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y,shuffle=False, test_size=0.2)

# (90 , 4, 1)

#2. 모델 구성 
model = Sequential()
# model.add(LSTM(140, activation='relu', input_shape=(4, 1)))
model.add(Conv1D(140, 2, padding='same', input_shape=(4,1)))
# model.add(MaxPool1D())
model.add(Conv1D(50, 2, padding='same'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(60))
model.add(Dense(25)) # Dense 는 2차원 되야함 2, 3차원 다받음
model.add(Dense(1)) # y.shape = (90, 1)

model.summary()


'''
# 3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'loss', patience=10, mode='auto')

model.fit(x, y, epochs=500, batch_size=1, verbose=1, callbacks=[early_stopping], shuffle=True, validation_split=0.2)

# 4. 평가, 예측
loss, mse = model.evaluate(x, y)

print("lose : ", loss)
print("mse : ", mse)


x_predict = dataset[90:96, :4]

print("x_predict : \n", x_predict)
print("x_predict.shape : ", x_predict.shape)

x_predict = x_predict.reshape(6, 4, 1)

print("x_predict : \n",x_predict)
print("x_predict.shape : ", x_predict.shape)

y_predict = model.predict(x_predict)

print("y_predict : \n", y_predict)
'''