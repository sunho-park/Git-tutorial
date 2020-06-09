import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# 데이터 불러오기
train_features = pd.read_csv('./data/dacon/comp2/train_features.csv')
train_target = pd.read_csv('./data/dacon/comp2/train_target.csv', index_col = 'id')
test_features = pd.read_csv('./data/dacon/comp2/test_features.csv')

# 데이터 형태 확인
print(f'train_features {train_features.shape}')  #  (1050000, 6)
print(f'train_target {train_target.shape}')      #  (2800, 4)
print(f'test_features {test_features.shape}')    #  (262500, 6)

print(train_target.head())

print(train_features.groupby(['id','Time']).count())


def preprocessing_KAERI(data) :
    
    # data: train_features.csv or test_features.csv
    # return: Random Forest 모델 입력용 데이터
 
    # 충돌체 별로 0.000116 초 까지의 가속도 데이터만 활용해보기 
    _data = data.groupby('id').head(375)
    print("_data : ", _data)

    # string 형태로 변환
    _data['Time'] = _data['Time'].astype('str')
    
    # Random Forest 모델에 입력 할 수 있는 1차원 형태로 가속도 데이터 변환
    _data = _data.pivot_table(index = 'id', columns = 'Time', values = ['S1', 'S2', 'S3', 'S4'])
    
    # column 명 변환
    _data.columns = ['_'.join(col) for col in _data.columns.values]
    
    return _data


train_features = preprocessing_KAERI(train_features)
test_features = preprocessing_KAERI(test_features)

print('train_features : \n', train_features)
print('test_features : \n', test_features)

print(f'train_features {train_features.shape}')  # (2800, 1500)  x
print(f'test_features {test_features.shape}')    # (700, 1500)   x_pred

# train test 분리

x_train, x_test, y_train, y_test = train_test_split(train_features, train_target, test_size=.2)

print('x_train.shape : ', x_train.shape)        #   (2240, 1500)
print('x_test.shape : ', x_test.shape)          #   (560, 1500)
print('y_train.shape : ', y_train.shape)        #   (2240, 4)
print('y_test.shape : ', y_test.shape)          #   (560, 4)


import sklearn
from sklearn.ensemble import RandomForestRegressor
# 2. 모델

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
parameters = {}

pipe = Pipeline([("scaler", MinMaxScaler()), ('RandomForestRegressor', RandomForestRegressor())])
# pipe = make_pipeline(MinMaxScaler(), SVC())

model = RandomizedSearchCV(pipe, parameters, cv=5)

model.fit(x_train, y_train)


# 평가 예측
score = model.score(x_test, y_test)
y_predict = model.predict(x_test)
print('y_predict.shape : ', y_predict.shape) 
print('y_predict : \n', y_predict)

R2 = r2_score(y_test, y_predict)
print('score : ', score)
print("R2 : ", R2) 

from sklearn.metrics import mean_absolute_error, mean_squared_error

print('mae : ', mean_absolute_error(y_test, y_predict))
print('mse : ', mean_squared_error(y_test, y_predict))

# submission 제출

y_pred = model.predict(test_features)
print('y_pred  : ', y_pred)

# 답안지 불러오기

submit = pd.read_csv('./data/dacon/comp2/sample_submission.csv')
print(submit.head())

# 답안지에 옮겨 적기
for i in range(4):
    submit.iloc[:,i+1] = y_pred[:,i]
print(submit.head())
submit.to_csv('./data/dacon/comp2/submission.csv', index = False)


# y_pred = model.predict(test_features)

