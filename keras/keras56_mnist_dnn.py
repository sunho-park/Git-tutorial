import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("x_train.shape : ", x_train.shape)  #(60000, 28, 28)
print("x_train : \n", x_train)
print("x_train[0]: ", x_train[0])
print("x_train[0].shape : ", x_train[0].shape) #(28, 28)


print("y_train.shape : ", y_train.shape) # (60000, ) 6만개 스칼라를 가진 벡터1개
print("y_train : \n", y_train)
print('y_train[0] : ', y_train[0])
print('y_train[0].shape : ', y_train[0].shape)


print("x_test.shape : ", x_test.shape)    #(10000, 28, 28) 
print("x_test : ", x_test)
print("x_test[0] : ", x_test[0])

 
print("y_test.shape : ", y_test.shape)   # (10000,)
print("y_test : ", y_test)
print("y_test[0] : ", y_test[0])




# print(x_train[0])
# plt.imshow(x_train[0], 'gray')
# plt.imshow(x_train[0])
# plt.show()


# 데이터 전처리

from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print("y_train : \n", y_train)
print("y_train.shape : ", y_train.shape) # (60000, 10) 
# 데이터 전처리 2. 정규화

x_train = x_train.reshape(60000, 784).astype('float32')/255                                                            
x_test = x_test.reshape(10000, 784).astype('float32')/255     


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, LSTM
from keras.layers import Dropout

model = Sequential()

model.add(Dense(32, activation='relu', input_shape=(28*28, )))  
model.add(Dense(64, activation='relu')) 
model.add(Dense(128, activation='relu')) 
model.add(Dropout(0.1))
model.add(Dense(256, activation='relu')) 
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu')) 
model.add(Dropout(0.4))
model.add(Dense(10, activation='softmax'))

model.summary()


# 컴파일, 훈련
from keras.callbacks import EarlyStopping, ModelCheckpoint

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

modelpath= './model/sample/mnist/check={epoch:02d}-{val_loss:.4f}.hdf5'

checkpoint = ModelCheckpoint(filepath=modelpath, monitor='val_loss', 
                             verbose=1,
                             save_best_only=True, save_weights_only=False )

es = EarlyStopping(monitor='loss', patience=5, mode='auto')

model.fit(x_train, y_train, epochs=30, batch_size=32, verbose=1, 
            validation_split=1/6, callbacks=[es, checkpoint])

model.save('./model/sample/mnist/mnist_model_save.h5')

model.save_weights('./model/sample/mnist/mnist_weight.h5')

# 평가, 예측
loss, acc = model.evaluate(x_test, y_test)

print("loss : ", loss)
print("acc : ", acc)

model.save('./model/sample/mnist/mnist_model_save.py')
