import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb
from sklearn.model_selection import KFold
import warnings
# warnings.fillterwarnings(action='ignore')

train = pd.read_csv('./data/dacon/comp1/train.csv')
test = pd.read_csv('./data/dacon/comp1/test.csv')
submission = pd.read_csv('./data/dacon/comp1/sample_submission.csv')

# print(train.head())

train.isna().sum().plot()
# plt.show()

test.isna().sum().plot()
# plt.show()

test = test.fillna(test.mean())
train = train.fillna(train.mean())

print(train.head)
print(test.head)

'''
train = train.interpolate(method='linear', axis=1)
test = test.interpolate(method='linear', axis=1)

train = train.fillna(0, inplace=True)
test = test.fillna(0, inplace=True)
'''

plt.figure(figsize=(4, 12))
ax = sns.heatmap(train.corr().loc['rho':'990_dst', 'hhb':].abs())
# plt.show()

x_train = train.loc[:, '650_dst':'990_dst']
y_train = train.loc[:, 'hhb':'na']

print(x_train.shape, y_train.shape)

def train_model(x_data, y_data, k=5):
    models=[]
    
    k_fold = KFold(n_splits=k, shuffle=True, random_state=123)

    for train_idx, val_idx in k_fold.split(x_data):
        
        x_train, y_train = x_data.iloc[train_idx], y_data[train_idx]
        x_val, y_val = x_data.iloc[val_idx], y_data[val_idx]

        d_train = xgb.DMatrix(data = x_train, label = y_train)
        d_val = xgb.DMatrix(data = x_val, label = y_val)

        wlist = [(d_train, 'train'), (d_val, 'eval')]


        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'mae',
            'seed' : 777
        }

        model = xgb.train(params=params, dtrain = d_train, num_boost_round=500, verbose_eval=500, evals=wlist)
        models.append(model)

    return models


models = {}
for label in y_train.columns:  # hhb, hbo2, ca, na 
    print('train column : ', label)
    models[label] = train_model(x_train, y_train[label])


for col in models:
    preds = []
    for model in models[col]:
        preds.append(model.predict(xgb.DMatrix(test.loc[:, '650_dst':])))
    pred = np.mean(preds, axis=0)

    submission[col] = pred

print(submission.head())

# submission.to_csv('Dacon.csv', index=False)

