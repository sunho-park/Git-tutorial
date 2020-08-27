import keras
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import np_utils

# 입력과 출력 지정하기 1
rows = 32    # 이미지의 세로 픽셀수
cols = 32    # 이미지의 가로 픽셀수
color = 3    # 이미지의 색 공간
in_shape = (rows, cols, color)
out_y = 3

# 사진 데이터 읽어 들이기 2
photos = np.load('myproject/photos.npz')
x = photos['x']     # image 
y = photos['y']     # label 

print('x.shape : ', x.shape)    #(600, 32, 32, 3)
print('y.shape : ', y.shape)    #(600,)

# 정규화 3
x = x.astype('float32') /255

# 레이블 데이터를 One-hot 인코딩 4
y = np_utils.to_categorical(y, out_y)   # y=(600, ) // to_categorical(600, 3)

# Train, Test 구분하기 5
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, train_size=0.8)

# CNN 모델 만들기 6
import sc4_cnn_model
model = sc4_cnn_model.get_model(in_shape, out_y)

# 훈련 7
hist = model.fit(x_train, y_train, batch_size=4, epochs=45, verbose=1, validation_split=0.25)

# 평가 8
loss, acc = model.evaluate(x_test, y_test,verbose=1)
print('정답률=', acc, '손실률=', loss)

# 훈련상태를 그래프로 그리기   9
# 정답률 그리기
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Accuracy')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# 손실률 추이 그리기
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Loss')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# 가중치 저장
model.save_weights('./myproject/photos-model-light_vgg.hdf5')

# acc이 1에 수렴하는 반면 val_Acc 이 낮게나옴 0.75정도