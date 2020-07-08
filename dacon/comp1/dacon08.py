import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from keras.callbacks import EarlyStopping
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.feature_selection import SelectFromModel
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns

# 1. 데이터 
dst_src = pd.read_csv('./data/dacon/comp1/train.csv', header=0, index_col=0)
x_predict = pd.read_csv('./data/dacon/comp1/test.csv', header=0, index_col=0)
submission = pd.read_csv('./data/dacon/comp1/sample_submission.csv', header=0, index_col=0)

print('dst_src.shape :', dst_src.shape)                   # (10000, 75) : x,y_train, test
print('x_predict.shape :', x_predict.shape)               # (10000, 71) : x_predict
print('submission.shape :', submission.shape)             # (10000,  4) : y_predict

only_dst = dst_src.filter(regex='_dst$', axis=1)
print(only_dst.shape) # (10000, 35)
# print(only_dst.head())
only_dst_test = x_predict.filter(regex='_dst$', axis=1)
print(only_dst_test.shape) #(10000, 35)
print('필터 후 only_dst 타입확인', type(only_dst))


# 결측치 보간법 처리
only_dst = only_dst.interpolate()
only_dst_test = only_dst_test.interpolate()

# nan 값 채우기
only_dst = only_dst.fillna(method='bfill')
only_dst_test = only_dst_test.fillna(method='bfill')
print('only_dst 타입확인 : ', type(only_dst)) # DataFrame

# src 데이터만 빼기
only_src = dst_src.filter(regex='_src$', axis=1)
# print(only_src.shape) # (10000, 35)
only_src_test = dst_src.filter(regex='_src$', axis=1)

# rho 계산
for col in only_dst:
    only_dst[col] = only_dst[col] * (dst_src['rho'] ** 2)
    only_dst_test[col] = only_dst_test[col] * (x_predict['rho']**2) 

# y 값 인덱싱 
y = dst_src.iloc[:, 71:]
print("인덱싱후y타입: ", type(y))
# y = y.values
print("numpy타입 변환 y타입 확인: ", type(y))

# src-dst gap 컬럼 추가
gap_feature_names=[]
for i in range(650, 1000, 10):
    gap_feature_names.append(str(i) + '_gap')

a=pd.DataFrame(np.array(only_src) - np.array(only_dst), columns=gap_feature_names, index=dst_src.index)
b=pd.DataFrame(np.array(only_src_test) - np.array(only_dst_test), columns=gap_feature_names, index=x_predict.index)

only_dst=pd.concat((only_dst, a), axis=1)
only_dst_test=pd.concat((only_dst_test, b), axis=1)