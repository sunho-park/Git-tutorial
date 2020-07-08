import sc4_cnn_model
import keras
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import cv2

# 입력과 출력 지정하기 1
rows = 32 # 이미지의 세로 픽셀수
cols = 32 # 이미지의 가로 픽셀수
color = 3 # 이미지의 색 공간
in_shape = (rows, cols, color)
out_y = 3

# 사진 데이터 읽어 들이기 2
photos = np.load('myproject/photos.npz')
x = photos['x']     # image 
y = photos['y']     # label 

# 정규화 3
x = x.astype('float32') /255 

# 레이블 데이터를 One-hot 벡터로 변환하기 4
y = keras.utils.np_utils.to_categorical(y, out_y)
print('x : ', x)
print(x.shape)
print('y : ', y)
print(y.shape)  # (600, 3)
# 학습 전용과 테스트 전용 구분하기 5
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, train_size=0.8)
print(len(x_test))
print(len(y_test))
#########################################################################
## OpenCV 이용하여 학습전용 데이터수 늘리기 ##
x_new = []                                                   # x_train.shape : (480, 32, 32, 3)
y_new = []                                                   # y_train.shape : (480,)
for i, xi in enumerate(x_train):                             # i : index(0~479), xi : value 
    yi = y_train[i]                                          # 0~479   
    for angle in range(-30, 30, 5):                          # [-30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25] = 12개
        # 회전 시키기
        center = (16, 16)                                    # 회전 중심
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0) # affin에 필요한 행렬 만들기 (회전 중심, 회전 각도, 배율)
        xi2 = cv2.warpAffine(xi, matrix, (32, 32))           # affin 변환 (변환하려는 이미지, 위에서 생성한 행렬, 사이즈)
        x_new.append(xi2)                                    # 각 이미지당 회전된 12개의 이미지       x : 480개 * 12 = 5760 
        y_new.append(yi)                                     # [1, 0, 0] or [0, 1, 0] or [0, 0, 1]  y : 480개 * 12 = 5760
        
        # 좌우 반전  [12] * 2 =24
        xi3 = cv2.flip(xi2, 1)                               # flip(이미지 데이터, 반전방향) 양수이므로 y축 반전
        x_new.append(xi3)
        y_new.append(yi)

# 이미지를 늘린 데이터를 학습 데이터로 사용하기
print('수량을 늘리기 전 = ', len(y_train))      # 600장 * 0.8 = 480 개
x_train = np.array(x_new)
y_train = np.array(y_new)
print('수량을 늘린 후 =', len(y_train))         # 480 * 24 = 11520 개
#########################################################################
# CNN 모델 만들기 6
model = sc4_cnn_model.get_model(in_shape, out_y)
model.save_weights('./myproject/photos-model-light.hdf5')

# 학습 실행하기
hist = model.fit(x_train, y_train, batch_size=32, epochs=60, verbose=1, validation_split=.25) 

# 모델 평가하기   8
loss, acc = model.evaluate(x_test, y_test, verbose=1)
print('정답률=', acc, '손실률=', loss)

# 학습상태를 그래프로 그리기   9
# 정답률 추이 그리기
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

model.save_weights('./myproject/photos-model-light.hdf5')
