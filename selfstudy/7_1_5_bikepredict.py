import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

bike_df = pd.read_csv('./selfstudy/bike-sharing-demand/bike_train.csv')
print(bike_df.shape)
print(bike_df.head())
print(bike_df.info())

# 문자열을 datetime 타입으로 변경
bike_df['datetime'] = bike_df.datetime.apply(pd.to_datetime)

# datetime 타입에서 년, 월, 일, 시간 추출
bike_df['year'] = bike_df.datetime.apply(lambda x : x.year)
bike_df['month'] = bike_df.datetime.apply(lambda x : x.month)
bike_df['day'] = bike_df.datetime.apply(lambda x : x.day)
bike_df['hour'] = bike_df.datetime.apply(lambda x : x.hour)

print(bike_df.head(3))

drop_columns = ['datetime', 'casual', 'registered']
bike_df.drop(drop_columns, axis=1, inplace=True)

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
    print('RMSE : {0:.3f}, RMSE:{1:.3F}, MAE:{2:.3F}'.format(rmsle_val, rmse_val, mae_val))

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso

y_target = bike_df['count']
x_features = bike_df.drop(['count'], axis=1, inplace=False)

print(x_features.shape) # (10886, 12)
print(y_target.shape)   # (10886, )

x_train, x_test, y_train, y_test = train_test_split(x_features, y_target, test_size=.3, random_state=0)

lr_reg = LinearRegression()
lr_reg.fit(x_train, y_train)
pred = lr_reg.predict(x_test)

evaluate_regr(y_test, pred)

def get_top_error_data(y_test, pred, n_tops=5):
    # DataFrame의 칼럼으로 실제 대여 횟수(count)와 예측값을 서로 비교할 수 있도록 생성.
    result_df = pd.DataFrame(y_test.values, columns=['real_count'])
    result_df['predicted_count'] = np.round(pred)
    result_df['diff']=np.abs(result_df['real_count']-result_df['predicted_count'])

    # 예측값과 실제 값이 가장 큰 데이터 순으로 출력
    print(result_df.sort_values('diff', ascending=False)[:n_tops])


get_top_error_data(y_test, pred, n_tops=5)

# y_target.hist()

# y 를 정규 분포로 바꿔줌
y_log_transform = np.log1p(y_target)
y_log_transform.hist()
plt.show()

# 타킷 칼럼인 count 값을 log1p로 로그 변환
y_target_log = np.log1p(y_target)

# 
