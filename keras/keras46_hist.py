# 45을 카피해서 46 복붙
# keras46_hist.py

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

#1. 데이터
a = np.array(range(1, 101)) # 1~10

size=5                       # time_steps=4 


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


         # [행, 열]  tip: 함수의 목적은 재사용 그러므로 함수를 변경할떄 고려해야함
x = dataset[ :, :4]  # or [:, 0:4],  [:] -> all
y = dataset[:, 4:5]  # or [:, 4]
print("x : \n", x)
print("y : \n", y)
x = x.reshape(96, 4, 1)
print("x.shape", x.shape)
print("y.shape", y.shape)

#2. 모델

from keras.models import load_model
model = load_model('./model/save_keras44.h5')

# from keras.layers import Dense
model.add(Dense(150, name='new1'))  # 이 줄이 dense_1로 들어가서 dense_1 이 2번사용되서 충돌이 일어남 
model.add(Dense(130, name='new2'))
model.add(Dense(30, name='new3'))
model.add(Dense(30, name='new4'))
model.add(Dense(30, name='new5'))
model.add(Dense(30, name='new6'))
model.add(Dense(1, name='new7'))     #남의 것 떙겨와도 hyper parameter 튜닝해야함. transfer learning (전이 학습)
model.summary()

#3. 훈련
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'loss', patience=5, mode='auto')

model.compile(loss='mse', optimizer='adam', metrics=['acc'])
hist = model.fit(x, y, epochs=100, validation_split=0.2, batch_size=1, verbose=1, callbacks=[early_stopping])

print(hist) 
print(hist.history.keys())

import matplotlib.pyplot as plt

plt.plot(hist.history['loss'])       # 한가지만 넣으면 y 값, plt.plot()의 갯수만큼 선이나옴
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('loss & acc')
plt.ylabel('loss, acc')
plt.xlabel('epoch')
plt.legend(['loss', 'val_loss', 'acc', 'val_acc'])
# plt.legend(['train loss', 'test loss', 'train acc', 'test acc'])
'''plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['acc', 'val_acc'])'''
plt.show()


'''
# 4. 평가 , 예측
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

'''