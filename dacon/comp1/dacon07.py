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

# print(dst_src.info())
# print(x_predict.info())

'''
print('rho 값의 분포: \n ', dst_src['rho'].value_counts())
print('rho 값의 분포: \n ', x_predict['rho'].value_counts())
print('hhb 값의 분포: \n ', dst_src['hhb'].value_counts())
print('hbo2 값의 분포: \n ', dst_src['hbo2'].value_counts())
print('ca 값의 분포: \n ', dst_src['ca'].value_counts(), )
print('na 값의 분포: \n ', dst_src['na'].value_counts())
print('dst_src src 값의 분포: \n ', dst_src['650_src'].value_counts())
print('x_predict src 값의 분포: \n ', x_predict['650_src'].value_counts())
print('dst_src dst 값의 분포: \n ', dst_src['650_dst'].value_counts())
print('x_predict dst 값의 분포: \n ', x_predict['650_dst'].value_counts())
print("--------------------------------------------------")
print('dst_src 650_src 최대값 : \n ', dst_src['650_src'].max())
print('dst_src 650_src 최소값 : \n ', dst_src['650_src'].min())
print('x_predict 최대값 : \n ', x_predict['650_src'].max())
print('x_predict src 최소값 : \n ', x_predict['650_src'].min())

print('dst_src 650_dst 최대값: \n ', dst_src['650_dst'].max())
print('dst_src 650_dst 최소값: \n ', dst_src['650_dst'].min())
'''
# ax = sns.barplot(x='650_dst', y='hhb', data=dst_src)
# plt.show()
print('dst 데이터 뺴기 전 :', type(dst_src))
# dst 데이터만 빼기
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

# 결측값 확인
# print(only_dst.isna().sum())
# print(only_dst_test.isna().sum())

print('only_dst 타입확인 : ', type(only_dst)) # DataFrame

# src 데이터만 빼기
only_src = dst_src.filter(regex='_src$', axis=1)
# print(only_src.shape) # (10000, 35)
only_src_test = dst_src.filter(regex='_src$', axis=1)

# rho 계산
for col in only_dst:
    only_dst[col] = only_dst[col] * (dst_src['rho'] ** 2)
    only_dst_test[col] = only_dst_test[col] * (x_predict['rho']**2) 

# print(only_dst.head(3))
# print(only_dst_test.shape)

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

print(only_dst_test.shape, only_dst_test.shape)
# model_scoring_cv(multi_model, Xtrain, Ytrain)

print("표준화 전 only_dst 타입 : ",type(only_dst))
print("표준화 전 only_dst_test 타입 : ",type(only_dst_test))
###############################################################
# 이산 푸리에 변환
'''
alpha_real=Xtrain[dst_list]
alpha_imag=Xtrain[dst_list]

beta_real=Xtest[dst_list]
beta_imag=Xtest[dst_list]

for i in tqdm(alpha_real.index):
    alpha_real.loc[i]=alpha_real.loc[i] - alpha_real.loc[i].mean()
    alpha_imag.loc[i]=alpha_imag.loc[i] - alpha_real.loc[i].mean()
    
    alpha_real.loc[i] = np.fft.fft(alpha_real.loc[i], norm='ortho').real
    alpha_imag.loc[i] = np.fft.fft(alpha_imag.loc[i], norm='ortho').imag

    
for i in tqdm(beta_real.index):
    beta_real.loc[i]=beta_real.loc[i] - beta_real.loc[i].mean()
    beta_imag.loc[i]=beta_imag.loc[i] - beta_imag.loc[i].mean()
    
    beta_real.loc[i] = np.fft.fft(beta_real.loc[i], norm='ortho').real
    beta_imag.loc[i] = np.fft.fft(beta_imag.loc[i], norm='ortho').imag
    
real_part=[]
imag_part=[]

for col in dst_list:
    real_part.append(col + '_fft_real')
    imag_part.append(col + '_fft_imag')
    
alpha_real.columns=real_part
alpha_imag.columns=imag_part
alpha = pd.concat((alpha_real, alpha_imag), axis=1)

beta_real.columns=real_part
beta_imag.columns=imag_part
beta=pd.concat((beta_real, beta_imag), axis=1)
#############################################################################

# 표준화 하면 datatype바뀜...
scaler = StandardScaler()
scaler.fit(only_dst)
only_dst = scaler.transform(only_dst)

only_dst_test = scaler.transform(only_dst_test)
print("표준화 후 only_dst 타입 : ",type(only_dst))
print("표준화 후 only_dst_test 타입 : ",type(only_dst_test))
# '''

# only_dst = pd.DataFrame(only_dst)
print('데이타프레임변환 : ', type(only_dst))

####

x_train ,x_test, y_train, y_test = train_test_split(only_dst, y, shuffle=True, random_state=66, test_size=.3)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
print('x_train타입 : ', type(x_train))
print('y_train타입 : ', type(y_train))

'''
#### multiOutputRegressor에서는 feature_importances_ 지원안함.#### 
model = MultiOutputRegressor(XGBRegressor())
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
y_pred = model.predict(x_test)
print('mae : ', mean_absolute_error(y_test, y_pred))
print('score : ', score)

y_predict = model.predict(only_dst_test)
print("y_predict.shape : ", y_predict.shape) # (10000, 4)

thresholds = np.sort(model.feature_importances_)
print(thresholds)

for thresh in thresholds:
    selection = SelectFromModel(model, threshold=thresh, prefit=True)

    select_x_train = selection.transfrom(x_train)

    selection_model = XGBRegressor()
    selection_model.fit(select_x_train, y_train)

    select_x_test = selection.transform(x_test)
    y_pred = selection_model.predict(select_x_test)

    score = r2_score(y_test, y_pred)

    print("thresh=%.3f, n=%d, R2: %.2F%%" %(thresh, select_x_train.shape[1], score*100.0))
'''
# xgb 사용
model = MultiOutputRegressor(XGBRegressor())
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
y_pred = model.predict(x_test)
print('mae : ', mean_absolute_error(y_test, y_pred))
print('score : ', score)

y_predict = model.predict(only_dst_test)

# score = model.score(x_test, y_test)
# y_pred = model.predict(only_dst_test)

print("y_predict.shape : ", y_predict.shape) # (10000, 4)
# mae:  1.6533088646045329 

'''
# Randomforest Regressore 사용 
model = RandomForestRegressor()
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
y_pred = model.predict(x_test)
y_predict = model.predict(only_dst_test)
print(model.feature_importances_)
print('score:', score)
print('mae : ', mean_absolute_error(y_test, y_pred))
print(y_predict.shape)

# mae = mean_absolute_error(y_test, y_predict)
# print('mae', mae)


# 'MultiOutputRegressor' object has no attribute 'best_estimator_'
model = XGBRegressor()

parameters = {'n_estimators': [800, 900],
               'learning_rate': [0.1, 0.3, 0.7], 
              # 'colsample_bytree': [0.6, 0.8], 
                'max_depth' : [4, 5, 6], 
                #"colsample_bylevel":[0.6, 0.7, 0.9]
                 }

model = GridSearchCV(XGBRegressor(), parameters, cv=5, n_jobs=-1, verbose=1)

multi_model=MultiOutputRegressor(model)

multi_model.fit(x_train, y_train)

print("=================================")
print(multi_model.best_estimator_)
print("=================================")
print(multi_model.best_params_)
print("=================================")

score = multi_model.score(x_test, y_test) # score는 evaluate
print('점수 :', score)
'''

a = np.arange(10000, 20000)
y_predict = pd.DataFrame(y_predict, a)
y_predict.to_csv('./data/dacon/comp1/sample_submission2.csv', index = True, header=['hhb','hbo2','ca','na'],index_label='id')

# '''

