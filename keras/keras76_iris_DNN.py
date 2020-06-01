
from sklearn.datasets import load_iris
import numpy as np

dataset = load_iris()
x = dataset.data
y = dataset.target

print('x : \n', x)
print('x.shape : ', x.shape) #(150, 4)
print('y : ', y)
print('y.shape : ', y.shape) #(150, ) - > (150, 1)



from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=False, test_size=0.1)

# 3 분류니깐 원핫인코딩
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print('y_train : \n', y_train)
print('y_test : ', y_test)


print('y_train.shape : ', y_train.shape)
print('y_test.shape : ', y_test.shape)
print('x_train.shape : ', x_train.shape)
print('x_test.shape : ', x_test.shape)


from keras.models import Sequential, Model
from keras.layers import Dense

# 모델 구성
model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(4, )))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax')) 

model.summary()

# 컴파일, 훈련
from keras.callbacks import EarlyStopping, ModelCheckpoint

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

modelpath='./model/sample/iris/check={epoch:02d}-{val_loss:.4f}.hdf5'

checkpoint = ModelCheckpoint(filepath=modelpath, monitor='val_loss',
                             verbose=1,
                             save_best_only=True, save_weights_only=False)
es = EarlyStopping(monitor='loss', patience=5, mode='auto')

model.fit(x_train, y_train, epochs=20, batch_size=1, 
            verbose=1, validation_split=0.1, callbacks=[es, checkpoint])

model.save('./model/sample/iris/iris_model_save.h5')
model.save_weights('./model/sample/iris/iris_weight.h5')

# 예측, 평가

loss, acc = model.evaluate(x_test, y_test)

print("loss : ", loss)
print("acc : ", acc)

# argmax ?

# loss :  0.10264165477206309
# acc :  0.9666666388511658

model.save('./model/sample/iris/76_iris_dnn.py')