from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential, Model, Input
from keras.applications import VGG16, VGG19

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

input_tensor = Input(shape=(32, 32, 3)) 
vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

# include_top은 원래 모델의 최후 전결합층을 사용할지 여부 False 일 경우 원래 ㅁ도델의 합성곱층의 특징 추출 부분만 사용
# 이후 층에는 스스로 작성한 모델을 추가할 수 있다.
# weights에 imagenet을 지정하면 Imagenet 에서 학습한 가중치를 사용하고 None을 지정하면 임의의 가중치를 사용

model = Sequential()
model.add(vgg16)
'''
model.add(Conv2D(32, kernel_size=3, padding='same', activation='relu', input_shape=(32, 32, 3)))
# model.add(Dropout(0.2))
model.add(Conv2D(32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))
model.add(Dropout(0.2))

model.add(Conv2D(64, kernel_size=3, padding='same', activation='relu'))
# model.add(Dropout(0.2))
model.add(Conv2D(64, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))
model.add(Dropout(0.2))
'''

model.add(Conv2D(128, kernel_size=3, padding='same', activation='relu'))
model.add(Dropout(0.3))
model.add(Conv2D(128, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
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
