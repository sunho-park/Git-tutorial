# keras113_L1.py 복사
# 과적합 피하기 2
# BatchNormalization  
# 1. 각 layer 에 들어가는 input 을 normalize 시킴으로써 layer의 학습을 가속하는데,
#    이 때 whitening 등의 방법을 쓰는 대신 각 mini-batch의 mean 과 variance를 구하여 normalize 한다.
# 2. 훨씬 큰 학습률을 사용할 수 있어 학습 속도를 개선
# 3. 시그모이드 함수나 하이퍼볼릭탄젠트 함수를 사용하더라도 기울기 소실 문제가 크게 개선
# 4. BatchNormalization을 적용하고 뒤에 Activation 사용

# 쓰는 이유 

# 다른 층들과 독립적. 각각의 층이 스스로 학습 진행할수록 도움. 
# 은닉층들의 정규화 필요
# 층이 깊어질수록 입력특성에서 정규화한 효과없어진다. 

# https://shuuki4.wordpress.com/2016/01/13/batch-normalization-%EC%84%A4%EB%AA%85-%EB%B0%8F-%EA%B5%AC%ED%98%84/
# https://blog.naver.com/ehdsnck/221769032128


# kernel initializer 는 레이어에서 Kernel(weight)와 Bias를 초기화 시키는 방법이다.

from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential, Model, Input

from keras.layers import Dense, LSTM, Conv2D
from keras.layers import Flatten, MaxPooling2D, Dropout, Activation, BatchNormalization # activation에서도 명시해줘야함
from keras.optimizers import Adam

import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = cifar10.load_data()


# 데이터 전처리 1. 원핫인코딩

# y_train = np_utils.to_categorical(y_train)
# y_test = np_utils.to_categorical(y_test)

# 정규화 / 피처를 늘린다 / 레귤러라리 제이션

# 데이터 전처리 2. 정규화
x_train = x_train.reshape(50000, 32, 32, 3).astype('float32')/255
x_test = x_test.reshape(10000, 32, 32, 3).astype('float32')/255

# 모델 구성 
from keras.regularizers import l1, l2, l1_l2

model = Sequential()
model.add(Conv2D(32, kernel_size=3, padding='same', input_shape=(32, 32, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(32, kernel_size=3, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(64, kernel_size=3, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(64, kernel_size=3, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(128, kernel_size=3, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(128, kernel_size=3, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))

model.add(Flatten())
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(10, activation='softmax'))

model.summary()

# 훈련
model.compile(optimizer=Adam(1e-4), loss='sparse_categorical_crossentropy', metrics=['acc']) 
# 따옴표'adam' 디폴트의 adam을 가져옴 / 1e-4 0.0001 / 원한인코딩안했을 경우 sparse_categorical_crossentropy 

hist = model.fit(x_train, y_train, epochs=20, batch_size=32, verbose=1, validation_split=0.3)

# 평가, 예측
loss_acc = model.evaluate(x_test, y_test, batch_size=32)

loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']

print("loss_acc : ", loss_acc)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)

plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')

plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')

plt.subplot(2, 1, 2)

plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])

plt.grid()
plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['acc', 'val_acc'])

plt.show()


