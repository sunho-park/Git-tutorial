# keras 85 복붙 


import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()


print("x_train : \n", x_train)
print("x_train.shape : ", x_train.shape)  #(60000, 28, 28)
print("x_train[0]: ", x_train[0])

print('y_train : ', y_train[0])


print("x_test.shape : ", x_test.shape)    #(10000, 28, 28) 
print("y_train.shape : ", y_train.shape) # (60000,) 6만개 스칼라를 가진 벡터1개
print("y_test.shape : ", y_test.shape)   # (10000,)

print("x_train[0].shape : ", x_train[0].shape) #(28, 28)

# print(x_train[0])
# plt.imshow(x_train[0], 'gray')
# plt.imshow(x_train[0])
# plt.show()


# 데이터 전처리 1. 원핫인코딩

from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print("y_train : \n", y_train)
print("y_train.shape : ", y_train.shape) # (60000, 10) https://ko.d2l.ai/chapter_crashcourse/linear-algebra.html (텐서 개념)

# 데이터 전처리 2. 정규화

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255                                                            
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255     

# 0(흰색) 255(완전진한검정) 
# reshape로 4차원 why cnn에 집어넣으려고 // x 각 픽셀마다 정수형태 0~255가 들어가 있음 min max 0~1 
# 255로 나누는 이유, 정규화                 y 는 0~9 까지
# x_train = x_train / 255   # (x - 최소) / (최대 - 최소)

from keras.models import load_model
model = load_model('./model/model_test01.h5') 

model.summary()


# 4. 평가, 예측

loss_acc = model.evaluate(x_test, y_test)
print('loss, acc : ', loss_acc)

# loss = hist.history['loss']
# val_loss = hist.history['val_loss']
# acc = hist.history['acc']
# val_acc = hist.history['val_acc']

# print("acc : ", acc)
# print("val_acc : ", val_acc)
print('loss_acc : ', loss_acc)

# 
'''
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
# acc :  [0.905875, 0.97020835, 0.9773125, 0.982125, 0.9840625, 0.9862083, 0.9869375, 0.98802084, 0.9890208, 0.9900208]
# val_acc :  [0.9749166369438171, 0.9821666479110718, 0.984499990940094, 0.9859166741371155, 0.9858333468437195, 0.9854166507720947, 0.9878333210945129, 0.9870833158493042, 0.9836666584014893, 0.984000027179718]
# loss_acc :  [0.05416520072426065, 0.982699990272522]
