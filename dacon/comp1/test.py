import pandas as pd                         # 데이터 분석 패키지
import numpy as np                          # 계산 패키지
import matplotlib.pyplot as plt             # 데이터 시각화 패키지
import seaborn as sns                       # 데이터 시각화 패키지
from xgboost import XGBClassifier, XGBRFRegressor                    
from sklearn.model_selection import KFold   # K-Fold CV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings(action='ignore') 
train = pd.read_csv('./data/dacon/comp1/train.csv', index_col=0)
test = pd.read_csv('./data/dacon/comp1/test.csv', index_col=0)
submission = pd.read_csv('./data/dacon/comp1/sample_submission.csv')

# print(train.head())

print('train.shape : ', train.shape)                    # (10000, 75)   : x_train, x_test, y_train, y_test
print('test.shape : ', test.shape)                      # (10000, 71)   : x_predict
print('submission.shape : ', submission.shape)          # (10000, 4)    : y_predict

# train = train.interpolate()

# print(train.head())
# print(train.tail())

# print(train.isnull().sum()[train.isnull().sum().values > 0])
train_dst = train.filter(regex='_dst$', axis=1).replace(0, np.NaN)
test_dst = test.filter(regex='_dst$', axis=1).replace(0, np.NaN) # 보간을 하기위해 결측값을 삭제한다.
# print('test_dst.head(1) : ', test_dst.head(1))

train_dst = train_dst.interpolate(methods='linear', axis=1)
test_dst = test_dst.interpolate(methods='linear', axis=1)

# 스팩트럼 데이터에서 보간이 되지 않은 값은 0으로 일괄 처리한다.
train_dst.fillna(0, inplace=True)
test_dst.fillna(0, inplace=True)

# print(test_dst.head())
# print(test_dst.shape)  # (10000, 35)

# print("rho : \n", train['rho'].value_counts())
# print("rho : \n", test['rho'].value_counts())

count_data = train.groupby('rho')['rho'].count()
print(count_data)

x = train_dst
# x = train_dst.drop(['hhb', 'hbo2', 'ca', 'na'], axis=0)
y = train[['hhb', 'hbo2', 'ca', 'na']]

# x = x.replace(0, np.NaN)
# x = x.interpolate(methods='linear', axis=1)

print(x.shape) # (10000, 71)
print(y.shape) # (10000, 4)

# print(x)
# print("y : ", y)

x = x.values
y = y.values


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2)

print(x_train.shape)    # (8000, 35)
print(y_train.shape)    # (8000, 4)
print(x_test.shape)     # (2000, 35)
print(y_test.shape)     # (2000, 4)   
# print(type(x_train))

# 모델
# model = DecisionTreeClassifier(max_depth=3)
# model = GradientBoostingClassifier()
# model = XGBClassifier()
model = XGBRFRegressor()
# model = RandomForestRegressor()
# model = RandomForestClassifier() 

# 훈련
model.fit(x_train, y_train)

score = model.score(x_test, y_test)

# acc = model.score()

print(model.feature_importances_)

print(x.shape[1]) #71
'''
x = x.values

def plot_feature_importances(model):
    n_features = x.shape[1]  
    
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')  #


    plt.yticks(np.arange(n_features), x.feature_names)     # 축에 구간 설정, 이름 표시
    plt.xlabel("Feature Importances")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)                                    # 축의 제한

plot_feature_importances(model)
plt.show()
'''
print("score : ", score)