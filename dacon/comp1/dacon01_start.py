import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


'''
id : 구분자
rho : 측정 거리 (단위: mm)
src : 광원 스펙트럼 (650 nm ~ 990 nm)
dst : 측정 스펙트럼 (650 nm ~ 990 nm)
hhb : 디옥시헤모글로빈 농도
hbo2 : 옥시헤모글로빈 농도
ca : 칼슘 농도
na : 나트륨 농도
'''

train = pd.read_csv('./data/dacon/comp1/train.csv', header=0, index_col=0, encoding='UTF8')
test = pd.read_csv('./data/dacon/comp1/test.csv', header=0, index_col=0, encoding='UTF8')

submission = pd.read_csv('./data/dacon/comp1/sample_submission.csv', header=0, index_col=0)

print('train.shape : ', train.shape)                    # (10000, 75)   : x_train, x_test, y_train, y_test
print('test.shape : ', test.shape)                      # (10000, 71)   : x_predict
print('submission.shape : ', submission.shape)          # (10000, 4)    : y_predict


# print(train.isnull().sum())

train = train.interpolate()  # 보건법 // 선형 보간 / 칼럼별 옆의 컬럼에 영향 미치지 않음
# print(train.isnull().sum())

train.fillna(method='bfill', inplace=True)

test = test.interpolate()
test.fillna(method='bfill', inplace=True)

print(train.head())
print(test.head())
# print(train.tail())
print(type(train))
print(type(submission))

train = train.values
test = test.values
submission = submission.values

print(type(train))

x = train[:, :71]
y = train[:, 71:]


print('test.shape : ', test.shape)
print('x.shape : ', x.shape)    # (10000, 71)
print('y.shape : ', y.shape)    # (10000, 4)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

from sklearn.preprocessing import MinMaxScaler, StandardScaler

scaler = StandardScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#shape 확인
print('x_train.shape : ', x_train.shape)     # (7000, 71)
print('x_test.shape : ', x_test.shape)       # (3000, 71)

print('y_train.shape : ', y_train.shape)     # (7000, 4)
print('y_test.shape : ', y_test.shape)       # (3000, 4)

# 모델
from keras.layers import Dense
from keras.models import Sequential

model=Sequential()

model.add(Dense(1024, input_dim=71, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(4, activation='relu'))

# 컴파일, 훈련

model.compile(loss='mae', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=25, batch_size=4, verbose=1, validation_split=0.3)


# 평가 예측

loss, mae = model.evaluate(x_test, y_test, batch_size=4)

x_predict = test

y_predict = model.predict(x_predict)

print('loss : ', loss)
print('y_predict : ', y_predict)
print('y_predict.shape : ', y_predict.shape)


print('MAE : ', mae)

# from sklearn.metrics import mean_absolute_error
# mean_absolute_error(y_test, y_predict)
# print('mae : ', mean_absolute_error(y_test, y_predict))


a = np.arange(10000, 20000)
y_predict = pd.DataFrame(y_predict,a)
y_predict.to_csv('./data/dacon/comp1/sample_submission.csv', index = True, header=['hhb','hbo2','ca','na'],index_label='id')


# y_predict = pd.DataFrame(y_predict)
# y_predict.to_csv('./data/dacon/comp1/submission.csv', index=False)

# MAE :  1.7439733743667603