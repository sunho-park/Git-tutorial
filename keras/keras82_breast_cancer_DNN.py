from sklearn.datasets import load_breast_cancer
import pandas as pd
dataset = load_breast_cancer()

x = dataset.data
y = dataset.target

print('x : \n', x)
print('x.shape : ', x.shape)  #(569, 30)

print('y : ', y)
print('y.shape : ', y.shape)  # (569,)

'''
print(dataset.DESCR)

df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df['target'] = dataset.target
print(df.head())

print(df.tail())

print(df.describe())

print(df.iloc[:,-1].value_counts())
'''

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=False, test_size=0.1)

print("x_train.shape : ", x_train.shape)
print("x_test.shape : ", x_test.shape)

print("y_train.shape : ", y_train.shape)          
print("y_test.shape : ", y_test.shape)   


from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv2D

# 2. 모델 구성
model=Sequential()

model.add(Dense(10, input_shape=(30, )))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(50, activation='relu'))                            
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='sigmoid'))            


# 3. 컴파일, 훈련
from keras.callbacks import EarlyStopping, ModelCheckpoint

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

modelpath = './model/sample/cancer/check={epoch:2d}-{val_loss:.4f}.hdf5'
checkpoint = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=1,
                                save_best_only=True, save_weights_only=False)

es = EarlyStopping(monitor='loss', patience=5, mode='auto')

model.fit(x_train, y_train, epochs=20, batch_size=1, verbose=1, validation_split=0.2, callbacks=[es, checkpoint])   
model.save('./model/sample/cancer/cancer_model_save.h5')
model.save_weights('./model/sample/cancer/cancer_weight.h5')

# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=1)
print("loss : ", loss)
print("acc : ", acc)

model.save('./model/sample/cancer/82_cancer_dnn.py')
