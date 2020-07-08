import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

card_df = pd.read_csv('./data/dacon/jeju/201901-202003.csv')
'''
print(card_df.shape)
print(card_df.head())
print(card_df.tail())
print(card_df.info())
print(card_df.isnull().sum())

card_df['REG_YYMM'] = card_df.datetime.apply(pd.to_datetime)

card_df['year'] = card_df.datetime.apply(lambda x : x.year)
card_df['month'] = card_df.datetime.apply(lambda x : x.month)

print(card_df.head(3))'''

# drop_columns = ['CARD_SIDO_NM', 'CARD_CCG_NM', 'STD_CLSS_NM', 'HOM_SIDO_NM', 'HOM_CCG_NM', 'AGE', 'SEX_CTGO_CD', 'FLC']
drop_columns = ['CARD_CCG_NM', 'HOM_SIDO_NM', 'HOM_CCG_NM', 'AGE', 'SEX_CTGO_CD', 'FLC']
card_df.drop(drop_columns, axis=1, inplace=True)

# rmsle, mse, rmse
from sklearn.metrics import mean_squared_error, mean_absolute_error

#log 값 변환시 NaN 등의 이슈로 log()가 아닌 log1p()를 이용해 RMSLE 계산
def rmsle(y, pred):
    log_y = np.log1p(y) 
    log_pred = np.log1p(pred)
    squared_error = (log_y - log_pred)**2
    rmsle = np.sqrt(np.mean(squared_error))
    return rmsle

# 사이킷런의 mean_square_error()를 이용해 RMSE 계산
def rmse(y, pred):
    return np.sqrt(mean_squared_error(y, pred))

# MSE, RMSE, RMSLE를 모두 계산
def evaluate_regr(y, pred):
    rmsle_val = rmsle(y, pred)
    rmse_val = rmse(y, pred)
    # MAE 는 사이킷런의 mean_absolute_error() 로 계산
    mae_val = mean_absolute_error(y, pred)
    print('RMSLE : {0:.3f}, RMSE:{1:.3F}, MAE:{2:.3F}'.format(rmsle_val, rmse_val, mae_val))

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso

y_target = card_df['AMT']
x_features = card_df.drop(['AMT'], axis=1, inplace=False)

x_train, x_test, y_train, y_test = train_test_split(x_features, y_target, test_size=.3, random_state=0)

lr_reg = LinearRegression()
lr_reg.fit(x_train, y_train)
pred = lr_reg.predict(x_test)

evaluate_regr(y_test, pred)



# 제출 파일 만들기
submission = pd.read_csv('./data/dacon/jeju/submission.csv', index_col=0)
submission = submission.drop(['AMT'], axis=1)
submission = submission.merge(card_df, left_on=['REG_YYMM', 'CARD_SIDO_NM', 'STD_CLSS_NM'], right_on=['REG_YYMM', 'CARD_SIDO_NM', 'STD_CLSS_NM'], how='left')
submission.index.name = 'id'
submission.to_csv('submission2.csv', encoding='utf-8-sig')
submission.head()

