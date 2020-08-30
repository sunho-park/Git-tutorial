import keras
from keras.models import Model, Sequential
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from PIL import Image
import numpy as np
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.utils import np_utils
from sklearn.model_selection import train_test_split


# 데이터 읽기
photos = np.load('./vggface/kface.npz')

x = photos['x']
y = photos['y']

print(x.shape)  # (201703, 224, 224, 3)
print(y.shape)  # (201703,)

# # 정규화
# x = x.astype('float32') / 255

# one hot 인코딩
# y = np_utils.to_categorical(y, 400)

# train test 구분하기
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, train_size=0.8)


model_A = load_model("./vggface/vggfacemodel.h5")
# model_A.summary()


model_B_on_A = keras.models.Sequential(model_A.layers[:-1])
model_B_on_A.add(keras.layers.Dense(400, activation='softmax'))
model_A.load_weights('./Darkhorseproject/matrix/vgg_face_weights.h5')


model_A_clone = keras.models.clone_model(model_A)
model_A_clone.set_weights(model_A.get_weights())


for layer in model_B_on_A.layers[:-1]:
    layer.trainable = False

model_B_on_A.compile(loss='sparse_categorical_crossentropy', optimizer="sgd", metrics=["accuracy"])


histroy = model_B_on_A.fit(x_train, y_train, epochs=4, validation_split=0.25)

for layer in model_B_on_A.layers[:-1]:
    layer.trainable = True

optimizer = keras.optimizers.SGD(lr=1e-4)    # 기본 학습률은 1e-2
model_B_on_A.compile(loss="sparse_categorical_crossentropy", optimizer = optimizer, metrics = ["accuracy"])

history = model_B_on_A.fit(x_train, y_train, epochs=16, validation_split=0.25)

loss, acc = model_B_on_A.evaluate(x_train, y_train)
print('정답률=', acc, '손실률=', loss)

