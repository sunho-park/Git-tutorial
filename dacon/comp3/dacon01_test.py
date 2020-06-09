import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from keras.layers import Dense, Conv1D, LSTM, Flatten, MaxPool1D
from keras.models import Sequential


# 데이터 불러오기
train_features = pd.read_csv('./data/dacon/comp2/train_features.csv')
train_target = pd.read_csv('./data/dacon/comp2/train_target.csv', index_col = 'id')
test_features = pd.read_csv('./data/dacon/comp2/test_features.csv')

# 데이터 형태 확인

x = train_features.drop('id', axis=1)
x_pred = test_features.drop('id', axis=1)

print(f'x {x.shape}')  #  (1050000, 6)
print(f'train_target {train_target.shape}')      #  (2800, 4)
print(f'x_pred {x_pred.shape}')    #  (262500, 6)

print(type(x))
print(x.shape)
print(train_target.head())


x = x.values
train_target = train_target.values
x_pred = x_pred.values

# id  까지 넣어보자
x = x.reshape(2800, 375, 5)  
x_pred = x_pred.reshape(700, 375, 5)
y = train_target                 # (2800, 4)

print(x.shape)                   # (2800, 375, 5)         x
print(x_pred.shape)              #  (700, 375, 6) 
print(train_target.shape)        # (2800, 4)              y  
print(y.shape)                   # (2800, 4)

x_train, x_test, y_train, y_test=train_test_split(x, y, shuffle=False, test_size=.2)

print(x_train.shape) # (2240, 375, 6)
print(x_test.shape)  # (560, 375, 6)
print(y_train.shape) # (2240, 4)
print(y_test.shape)  # (560, 4)

# 모델 구성
model = Sequential()
model.add(Conv1D(256, 2, activation='relu', input_shape=(375, 5)))
model.add(MaxPool1D())
model.add(Conv1D(128, 2, activation='relu'))
model.add(MaxPool1D())
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(4, activation='relu'))

# 컴파일 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=100, batch_size=64, verbose=1, validation_split=0.25)

loss, mse = model.evaluate(x_test, y_test)

y_predict = model.predict(x_pred)

print("y_predict : ", y_predict)
print("y_predict.shape : ", y_predict.shape) 

print("loss : ", loss)
print("mse : ", mse)

# loss :  36303.67912946428
# mse :  36303.68359375

# 답안지 불러오기

submit = pd.read_csv('./data/dacon/comp2/sample_submission.csv')
print(submit.head())

# 답안지에 옮겨 적기
for i in range(4):
    submit.iloc[:,i+1] = y_predict[:,i]
print(submit.head())


submit.to_csv('./data/dacon/comp2/submission.csv', index = False)
