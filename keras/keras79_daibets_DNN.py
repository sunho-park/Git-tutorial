from sklearn.datasets import load_diabetes
import numpy as np

from sklearn import model_selection
from sklearn import metrics
import pandas as pd

dataset = load_diabetes()
x = dataset.data
y = dataset.target

print('x : \n', x)
print('x.shape : ', x.shape) # (442, 10)
print('y : ', y)
print('y.shape : ', y.shape) #(442,)
from sklearn import datasets


# 정규화
from sklearn.preprocessing import MinMaxScaler, StandardScaler, robust_scale, normalize

scaler = MinMaxScaler()      

scaler.fit(x)
x = scaler.transform(x) 

'''
# PCA
from sklearn.decomposition import PCA

pca = PCA(n_components=column)
x=pca.fit_transform(x)
'''


print(dataset.DESCR)

df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df['target'] = dataset.target
print(df.head())

print(df.tail())

print(df.describe())

print(df.iloc[:,-1].value_counts())


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=False, test_size=0.1)

print("x_train.shape : ", x_train.shape)
print("x_test.shape : ", x_test.shape)

print("y_train.shape : ", y_train.shape)          
print("y_test.shape : ", y_test.shape)   


from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense(10, input_shape = (10, )))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(1))

# 컴파일, 훈련
from keras.callbacks import EarlyStopping, ModelCheckpoint

model.compile(loss='mse', optimizer='adam', metrics=['mse'])
modelpath = './model/sample/diabetes/check={epoch:02d}-{val_loss:.4f}.hdf5'

checkpoint = ModelCheckpoint(filepath=modelpath, monitor='val_loss',
                                verbose=1,
                                save_best_only=True, save_weights_only=False)

es = EarlyStopping(monitor='loss', patience=5, mode='auto')

model.fit(x_train, y_train, epochs=20, batch_size=1, verbose=1, validation_split=0.1, callbacks=[es, checkpoint])

model.save('./model/sample/diabetes/diabetes_model_save.h5')
model.save_weights('./model/sample/diabetes/diabetes_weight.h5')
# 훈련 평가

loss, mse = model.evaluate(x_test, y_test)

print("loss : ", loss)
print("mse : ", mse)

model.save('./model/sample/diabetes/79_diabetes_dnn.py')