import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout
from xgboost import XGBRegressor, plot_importance
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from keras.callbacks import EarlyStopping
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.feature_selection import SelectFromModel


#1. data
train = pd.read_csv('./data/dacon/comp1/train.csv', index_col= 0 , header = 0)
test = pd.read_csv('./data/dacon/comp1/test.csv', index_col= 0 , header = 0)
submission = pd.read_csv('./data/dacon/comp1/sample_submission.csv', index_col= 0 , header = 0)

print('train.shape: ', train.shape)              # (10000, 75)  = x_train, test
print('test.shape: ', test.shape)                # (10000, 71)  = x_predict
print('submission.shape: ', submission.shape)    # (10000, 4)   = y_predict

train = train.interpolate()                       
test = test.interpolate()

train.fillna(method='bfill', inplace=True)
test.fillna(method='bfill', inplace=True)

x = train.iloc[:, :71]
y = train.iloc[:, 71:]

x1 = x.values
y1 = y.values
x_pred = test.values

print(x.shape)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x1, y1, train_size = 0.8, random_state=66)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

#2. feature_importance
xgbr = XGBRegressor()
model = MultiOutputRegressor(xgbr)
model.fit(x_train, y_train)

for i in range(len(model.estimators_)):
    threshold = np.sort(model.estimators_[i].feature_importances_)

    for thresh in threshold:
        selection = SelectFromModel(model.estimators_[i], threshold=thresh, prefit=True)

        param={
            'n_estimators' : [100, 200, 400],
            'learning_rate' : [0.05, 0.07, 0.09, 1],
            'max_depth' : [4, 5, 6]
        }

        grid = RandomizedSearchCV(XGBRegressor(), param, cv=5, n_jobs = -1)
        
        select_x_train = selection.transform(x_train)
        selection_model = MultiOutputRegressor(grid)
        selection_model.fit(select_x_train, y_train)
        
        select_x_test = selection.transform(x_test)
        y_pred = selection_model.predict(select_x_test)
        mae = mean_absolute_error(y_test, y_pred)

        score = r2_score(y_test, y_pred)
        print('Thresh=%.3f, n=%d, R2: %.2f%%, MAE: %.3f' %(thresh, select_x_train.shape[1], score*100.0, mae))

        select_x_pred = selection.transform(x_pred)
        y_predict = selection_model.predict(select_x_pred)

        a = np.arange(10000, 20000)
        submission = pd.DataFrame(y_predict, a)
        submission.to_csv('./data/dacon/comp1/sub_XG%d_%.5f.csv'%(i, mae),index = True, header=['hhb','hbo2','ca','na'],index_label='id')