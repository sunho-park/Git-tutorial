from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential, Model

from keras.layers import Dense, LSTM, Conv2D, Input
from keras.layers import Flatten, MaxPooling2D, Dropout

import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print('x_train[0] : ', x_train[0])
print('y_train[0] : \n', y_train[0])

print("y_train : \n", y_train)
print("y_test : ", y_test)
print("y_train.shape : ", y_train.shape)
print("y_test.shape : ", y_test.shape)


# plt.imshow(x_train[0])
# plt.show()

from keras.utils import np_utils

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)


print("y_train : ", y_train)
print("y_test : ", y_test)

x_train = x_train.reshape(50000, 32*32*3).astype('float32')/255
x_test = x_test.reshape(10000, 32*32*3).astype('float32')/255

input1 = Input(shape=(32*32*3,)) # 32*32*3 = 3072
dense1 = Dense(16, activation='relu')(input1)
dense1 = Dense(32, activation='relu')(dense1)

output1 = Dense(64, activation='relu')(dense1)
output1 = Dense(128, activation='relu')(dense1)
output1 = Dense(256, activation='relu')(dense1)
output1 = Dense(512, activation='relu')(dense1)
output1_1 = Dense(10, activation='softmax')(output1)


model = Model(inputs= input1, outputs = output1_1)

model.summary()

# 컴파일 훈련

from keras.callbacks import EarlyStopping, ModelCheckpoint
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
modelpath = './model/sample/cifar10/check={epoch:02d}-{val_loss:.4f}.hdf5'

checkpoint = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=1,
                            save_best_only=True, save_weights_only=False)

es = EarlyStopping(monitor='loss', patience=5, mode='auto')

model.fit(x_train, y_train, epochs=15, batch_size=128, verbose=1 ,validation_split=0.2, callbacks=[es, checkpoint])
model.save('./model/sample/cifar10/cifar10_model_save.h5')
model.save_weights('./model/sample/cifar10/cifar10_weight.h5')

# 평가 예측

loss, acc = model.evaluate(x_test, y_test)

print("loss : ", loss)
print("acc : ", acc)

model.save('./model/sample/cifar10/61_cifar10_dnn.py')