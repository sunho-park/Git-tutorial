import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

cust_df = pd.read_csv("./selfstudy/santander/train_santander.csv", encoding='latin-1')
print('dataset shape :', cust_df.shape)   # (76020, 371)

cust_df['var3'].replace(-999999, 2, inplace=True)
cust_df.drop('ID', axis=1 , inplace=True)

# 피처 세트와 레이블 세트 분리. 레이블 칼럼은 DataFrame의 맨 마지막에 위치해 칼럼 위치 -1로 분리
x_feature = cust_df.iloc[:, :-1]
y_labels = cust_df.iloc[:, -1]
print('피처 데이터 shape {0}'.format(x_feature.shape)) # (76020, 369) drop ID 해서 371 - 2

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_feature, y_labels, test_size=.2, random_state=0)

from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

lgbm_clf = LGBMClassifier(n_estimators=1000, num_leaves=32, sumsample=0.8, min_child_samples=100, max_depth=128)

evals = [(x_test, y_test)]
lgbm_clf.fit(x_train, y_train, early_stopping_rounds=100, eval_metric="auc", eval_set=evals, verbose=True)

lgbm_roc_score = roc_auc_score(y_test, lgbm_clf.predict_proba(x_test)[:, 1], average='macro')
print('ROC AUC:{0:.4f}'.format(lgbm_roc_score))
print('lgbm_clf.predict_proba(x_test)', lgbm_clf.predict_proba(x_test))
print('lgbm_clf.predict_proba(x_test)[:, 1]', lgbm_clf.predict_proba(x_test)[:, 1])
print('y_test', y_test)

from sklearn.model_selection import GridSearchCV
'''
# 하이퍼 파라미터 테스트의 수행속도를 향상시키기 위해 n_estimators를 200 감소
from sklearn.model_selection import GridSearchCV
lgbm_clf = LGBMClassifier(n_estimators=200)
params = {'num_leaves':[32, 64], 'max_depth':[128, 160], 'min_child_samples':[60, 100], 'subsample':[0.8, 1]}

# cv=3
gridcv = GridSearchCV(lgbm_clf, param_grid=params, cv=3)
gridcv.fit(x_train, y_train, early_stopping_rounds=30, eval_metric="auc", eval_set=[(x_train, y_train), (x_test, y_test)])

print('GridSearchCV 최적 파라미터:', gridcv.best_params_)
lgbm_roc_score = roc_auc_score(y_test, gridcv.predict_proba(x_test)[:, 1], average='macro')
print('ROC AUC :{0:.4f}'.format(lgbm_roc_score))'''

from lightgbm import plot_importance
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(10, 8))
plot_importance(lgbm_clf, ax=ax, max_num_features=20, height=.4)
plt.show()

