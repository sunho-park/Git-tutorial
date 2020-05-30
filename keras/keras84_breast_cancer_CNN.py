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

print("x_train.shape : ", x_train.shape)  # (512, 30)
print("x_test.shape : ", x_test.shape)    # (57, 30)

print("y_train.shape : ", y_train.shape)  #(512, )
print("y_test.shape : ", y_test.shape)    #(57, )

x_train = x_train.reshape(512, 30, 1, 1)
x_test = x_test.reshape(57, 30, 1, 1)



from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv2D, Flatten

# 2. 모델 구성
model=Sequential()

model.add(Conv2D(10, (1, 1), input_shape=(30, 1, 1)))
model.add(Conv2D(10, (1, 1), activation='relu'))
model.add(Conv2D(10, (1, 1), activation='relu'))
model.add(Conv2D(50, (1, 1), activation='relu'))                            
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))            


# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=20, batch_size=1, verbose=1)   


# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=1)
print("loss : ", loss)
print("acc : ", acc)
