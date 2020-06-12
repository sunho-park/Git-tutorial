# keras 90 복붙 

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.callbacks import EarlyStopping, ModelCheckpoint


x_train = np.load('./data/mnist_train_x.npy')
y_train = np.load('./data/mnist_train_y.npy')

x_test = np.load('./data/mnist_test_x.npy')
y_test = np.load('./data/mnist_test_y.npy')

print(x_train.shape)
print(y_train.shape)

print(x_test.shape)
print(y_test.shape)


# 데이터 전처리 1. 원핫인코딩

from keras.utils import np_utils

y_test = np_utils.to_categorical(y_test)
y_train = np_utils.to_categorical(y_train)

print("y_train : \n", y_train)
print("y_train.shape : ", y_train.shape) # (60000, 10) 

# 데이터 전처리 2. 정규화

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255                                                            
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255     

# 0(흰색) 255(완전진한검정) 
# reshape로 4차원 why cnn에 집어넣으려고 // x 각 픽셀마다 정수형태 0~255가 들어가 있음 min max 0~1 
# 255로 나누는 이유, 정규화                 y 는 0~9 까지
# x_train = x_train / 255   # (x - 최소) / (최대 - 최소)

'''
# 모델
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

model = Sequential()
model.add(Conv2D(32, (1, 1), input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3))) 
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3))) 
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(10, activation='softmax'))

# model.save('./model/model_test01.h5')


# 3, 컴파일 훈련

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc']) # metrics=['accuracy'] acc, accuracy 통일해야함

modelpath = './model/check={epoch:02d}-{val_loss:.4f}.hdf5'

checkpoint = ModelCheckpoint(filepath=modelpath, monitor='val_loss', 
                             verbose=1,
                             save_best_only=True, save_weights_only=False )

es = EarlyStopping(monitor= 'loss', patience=20)
hist = model.fit(x_train, y_train, epochs=30, batch_size=128, verbose=1, validation_split=0.2, callbacks = [es, checkpoint])

model.save('./model/model_test01.h5')
'''

from keras.models import load_model
model = load_model('./model/check=06-0.0464.hdf5')

# 4. 평가, 예측

loss_acc = model.evaluate(x_test, y_test)
print('loss_acc : ', loss_acc)

'''
loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']

print("acc : ", acc)
print("val_acc : ", val_acc)
print('loss_acc : ', loss_acc)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))  # 가로, 세로 길이 같음

plt.subplot(2, 1, 1)  # 2행1열의 그림을 그리겠다. 2행 1열의 1번째꺼 

plt.plot(hist.history['loss'], marker='.', c = 'red', label = 'loss')       # 한가지만 넣으면 y 값, plt.plot()의 갯수만큼 선이나옴
plt.plot(hist.history['val_loss'], marker='.', c = 'blue', label='val_loss') # 레전드의 라벨 명, c 는 선의 색깔

plt.grid()              # 격자형태
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
# plt.legend(['loss', 'val_loss']) # plot이랑 match 됨
plt.legend(loc='upper right') #명시 하지 않으면 빈자리 찾아서 알아서 넣어줌,  

plt.subplot(2, 1, 2) 

plt.plot(hist.history['acc'])     
plt.plot(hist.history['val_acc'])

plt.grid() 
plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['acc', 'val_acc'])

plt.show()
'''