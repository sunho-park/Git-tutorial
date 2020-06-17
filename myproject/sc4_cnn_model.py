import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

# CNN 모델 구성
def def_model(in_shape, out_y):
    model=Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=in_shape))

    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(out_y, activation='softmax'))
    return model


# 컴파일 하고 모델 반환하기
def get_model(in_shape, out_y):
    model = def_model(in_shape, out_y)
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])   
    return model
