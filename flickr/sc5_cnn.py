# from flickr.sc4_cnn_model 
# import flickr.sc4_cnn_model
import sc4_cnn_model
import keras
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

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

# 읽어 들인 데이터를 3차원 배열로 변환하기   3
x = x.reshape(-1, im_rows, im_cols, im_color)
x = x.astype('float32') /255 # 정규화

# 레이블 데이터를 One-hot 벡터로 변환하기 4
y =keras.utils.np_utils.to_categorical(y.astype('int32'), nb_classes)

# 학습 전용과 테스트 전용 구분하기 5
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)

# CNN 모델 만들기 6
model = sc4_cnn_model.get_model(in_shape, nb_classes)

# 학습 실행하기

hist = model.fit(x_train, y_train, batch_size=32, epochs=100, verbose=1, validation_data=(x_test, y_test))

# 모델 평가하기   8
score =model.evaluate(x_test, y_test, verbose=1)
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