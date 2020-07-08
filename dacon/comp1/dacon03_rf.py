import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout
from xgboost import XGBRegressor, plot_importance
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from keras.callbacks import EarlyStopping
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

#1. data
train = pd.read_csv('./data/dacon/comp1/train.csv', index_col= 0 , header = 0)
test = pd.read_csv('./data/dacon/comp1/test.csv', index_col= 0 , header = 0)
submission = pd.read_csv('./data/dacon/comp1/sample_submission.csv', index_col= 0 , header = 0)

print('train.shape: ', train.shape)              # (10000, 75)  = x_train, test
print('test.shape: ', test.shape)                # (10000, 71)  = x_predict
print('submission.shape: ', submission.shape)    # (10000, 4)   = y_predict


trian = train.interpolate(axis = 0)
test= test.interpolate(axis = 0)

train = train.fillna(train.mean())
test = test.fillna(test.mean())


x = train.iloc[:, :71]                           
y = train.iloc[:, -4:]
print(x.shape)                                   # (10000, 71)
print(y.shape)                                   # (10000, 4)

x = x.values
y = y.values
x_pred = test.values

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size =0.8,
                                                    shuffle = True, random_state = 66)


# feature_importance
model = RandomForestRegressor()
model.fit(x_train, y_train)
score = model.score(x_test, y_test)


threshold = np.sort(model.feature_importances_)

for thresh in threshold:
    selection = SelectFromModel(model, threshold = thresh, prefit = True)
        
    parameter = {
            'n_estimators': [100, 200, 400],
            'max_features': ["auto", "sqrt", "log2"],
            'min_samples_leaf': [1, 2, 5, 10],
            'max_depth': [5, 10, 100, 200, 500]
        }
    
    search = GridSearchCV(RandomForestRegressor(), parameter, cv =5)

    select_x_train = selection.transform(x_train)

    selection_model = RandomForestRegressor(search, n_jobs = -1)
    selection_model.fit(select_x_train, y_train )
        
    select_x_test = selection.transform(x_test)

    y_pred = selection_model.predict(select_x_test)

    mae = mean_absolute_error(y_test, y_pred)
    score =r2_score(y_test, y_pred)
    print("Thresh=%.3f, n = %d, R2 : %.2f%%, MAE : %.3f"%(thresh, select_x_train.shape[1], score*100.0, mae))
 
    select_x_pred = selection.transform(x_pred)
    y_pred = selection_model.predict(select_x_pred)

    # submission
    a = np.arange(10000, 20000)
    submission = pd.DataFrame(y_pred, a)
    # submission.to_csv('./dacon/comp1/sub_XG%i_%.5f.csv'%(i, mae),index = True, header=['hhb','hbo2','ca','na'], index_label='id')
    submission.to_csv('./dacon/comp1/sample_submission.csv', index = True, header=['hhb','hbo2','ca','na'], index_label='id')
        