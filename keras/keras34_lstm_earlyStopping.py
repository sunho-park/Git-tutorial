#keras23_emesemble3.py
#앙상블 
#2개의 모델이 들어가서 하나의 모델이 나오게 
#earlyStopping 적용하시오

from numpy import array
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input

# 1. 데이터

x1 = array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6],
            [5, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 10],
            [9, 10, 11], [10, 11, 12],
            [20, 30, 40],[30, 40, 50], [40, 50, 60]])                #(13, 3)

x2 = array([[10, 20, 30], [20, 30, 40], [30, 40, 50], [40, 50, 60],
            [50, 60, 70], [60, 70, 80], [70, 80, 90], [80, 90, 100],
            [90, 100, 110], [100, 110, 120],
            [2, 3, 4],[3, 4, 5], [4, 5, 6]])                          #(13, 3)    

y = array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 50, 60, 70])       #(13, )


print("y.shape : ", y.shape)   #(13, )
print("x : \n", x1)

x1 = x1.reshape(13, 3, 1) 
x2 = x2.reshape(13, 3, 1)

print("x.shape : ", x1.shape)   #(13, 3)
print("x.shape : ", x2.shape)   #(13, 3)

# x = x.reshape(x.shape[0], x.shape[1], 1)


print("x : \n ", x1)
print("x.shape", x1.shape)
# 모델구성
#model.add(LSTM(10, activation='relu', input_shape=(3, 1))) # (none, 3, 1)
# model.add(LSTM(10, input_length=3, input_dim=1))
input1 = Input(shape=(3, 1))
dense1_1 = LSTM(10)(input1)
dense1_1 = Dense(100)(dense1_1)
dense1_1 = Dense(100)(dense1_1)
dense1_1 = Dense(100)(dense1_1)
dense1_1 = Dense(100)(dense1_1)
dense1_2 = Dense(100)(dense1_1)
dense1_2 = Dense(4)(dense1_2)

input2 = Input(shape=(3,1))
dense2_1= LSTM(10)(input2)
dense2_1 = Dense(100)(dense2_1)
dense2_1 = Dense(100)(dense2_1)
dense2_1 = Dense(100)(dense2_1)
dense2_1 = Dense(100)(dense2_1)
dense2_2 = Dense(1)(dense2_1)

from keras.layers.merge import concatenate
merge1 = concatenate([dense1_2, dense2_2])

middle1 = Dense(15)(merge1)
middle1 = Dense(5)(middle1)
middle1 = Dense(7)(middle1)

output1 = Dense(30)(middle1)
output1_1 = Dense(7)(output1)
output1_1 = Dense(1)(output1_1)

model = Model(inputs=[input1, input2], outputs=output1_1)
model.summary()


# 3. 실행
model.compile(loss='mse', optimizer='adam')

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'loss', patience=50, mode='min')
model.fit([x1, x2], y, epochs=1500, verbose=1, callbacks=[early_stopping])

x1_predict = array([55, 65, 75])  
x2_predict = array([65, 75, 85])
x1_predict = x1_predict.reshape(1, 3, 1)
x2_predict = x2_predict.reshape(1, 3, 1)

print("x1_predict : \n", x1_predict)
print("x2_predict : \n", x2_predict)

print("x1_predict.shape", x1_predict.shape)
print("x2_predict.shape", x2_predict.shape)

print("---------------------------------------------------------------")



y_predict = model.predict([x1_predict, x2_predict])

print("y_predict : ", y_predict)




