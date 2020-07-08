import sc4_cnn_model as cnn
import keras
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import cv2


# 입력과 출력 지정하기 1
im_rows = 32 # 이미지의 세로 픽셀수
im_cols = 32 # 이미지의 가로 픽셀수
im_color = 3 # 이미지의 색 공간
in_shape = (im_rows, im_cols, im_color)
nb_classes = 3

# 사진 데이터 읽어 들이기 2
photos = np.load('flickr/photos.npz')
x = photos['x']
y = photos['y']

print('x.dtype : ', x.dtype)    # uint8
print('y.dtype : ', y.dtype)    # int32

print('type_x : ', type(x))     # numpy.ndarray
print('type_y: ', type(y))      # numpy.ndarray

# 읽어 들인 데이터를 3차원 배열로 변환하기   3
x = x.reshape(-1, im_rows, im_cols, im_color)
x = x.astype('float32') /255 # 정규화
print('x.dtype : ', x.dtype) # float32

# 레이블 데이터를 One-hot 벡터로 변환하기 4
y = keras.utils.np_utils.to_categorical(y.astype('int32'), nb_classes)
print('y.dtype : ', y.dtype) # float32

# 학습 전용과 테스트 전용 구분하기 5
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)


#########################################################################

# 학습전용 데이터수 늘리기
x_new = []
y_new = []
for i, xi in enumerate(x_train):
    yi = y_train[i]
    for ang in range(-30, 30, 5): #
        # 회전 시키기
        center = (16, 16) # 회전 중심
        mtx = cv2.getRotationMatrix2D(center, ang, 1.0) 
        xi2 = cv2.warpAffine(xi, mtx, (32, 32))
        x_new.append(xi2)
        y_new.append(yi)
        # 좌우 반전 
        xi3 = cv2.flip(xi2, i)
        x_new.append(xi3)
        y_new.append(yi)

# 이미지를 늘린 데이터를 학습 데이터로 사용하기
print('수량을 늘리기 전 = ', len(y_train))      # 240 개
x_train = np.array(x_new)
y_train = np.array(y_new)
print('수량을 늘린 후 =', len(y_train))         # 5760 개 = 240 * 24
#########################################################################

# CNN 모델 만들기 6
model = cnn.get_model(in_shape, nb_classes)

# 학습 실행하기

hist = model.fit(x_train, y_train, batch_size=32, epochs=40, verbose=1, validation_data=(x_test, y_test))

# 모델 평가하기   8
score=model.evaluate(x_test, y_test, verbose=1)
print('정답률=', score[1], '손실률=', score[0])

loss, acc = model.evaluate(x_test, y_test,verbose=1)
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

model.save_weights('./flickr/photos-model-light.hdf5')

