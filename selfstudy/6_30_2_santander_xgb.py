import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

cust_df = pd.read_csv("./selfstudy/santander/train_santander.csv", encoding='latin-1')
print('dataset shape :', cust_df.shape)   # (76020, 371)
print(cust_df.head(3))
print(cust_df.info())

print(cust_df['TARGET'].value_counts())
unsatisfied_cnt = cust_df[cust_df['TARGET']==1].TARGET.count()

total_cnt = cust_df.TARGET.count()
print('unsatisfied 비율은 {0:.2f}'.format((unsatisfied_cnt/total_cnt)))

print(cust_df.describe())
print(cust_df.var3.value_counts()[:10])

cust_df['var3'].replace(-999999, 2, inplace=True)
cust_df.drop('ID', axis=1 , inplace=True)

# 피처 세트와 레이블 세트 분리. 레이블 칼럼은 DataFrame의 맨 마지막에 위치해 칼럼 위치 -1로 분리
x_feature = cust_df.iloc[:, :-1]
y_labels = cust_df.iloc[:, -1]
print('피처 데이터 shape {0}'.format(x_feature.shape)) # (76020, 369) drop ID 해서 371 - 2
'''

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_feature, y_labels, test_size=.2, random_state=0)

train_cnt = y_train.count()
test_cnt = y_test.count()
# print('train_cnt:', train_cnt) # 60816
# print('train_cnt.shape:', y_train.shape) #(60816,)
print('train_value.count() :\n', y_train.value_counts()) # 0 58442 1 2374
print('학습 세트 Shape:{0}, 테스트 세트 Shape:{1}'.format(x_train.shape, x_test.shape))
print(y_train.value_counts()/train_cnt)

print('\n 테스트 세트 레이블 값 분포 비율')
print(y_test.value_counts()/test_cnt)

from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

# n_estimator는 500으로, random_state는 예제 수행 시마다 동일 예측 결과를 위해 설정.
xgb_clf = XGBClassifier(n_estimators=1000, random_state=156, learning_rate=0.02, max_depth=6, min_child_weight=3, colsample_bytree=0.5, reg_alpha=0.03)

# 성능 평가 지표를 auc로, 조기 중단 파라미터는 100으로 설정하고 학습 수행.
xgb_clf.fit(x_train, y_train, early_stopping_rounds=200, eval_metric="auc", eval_set=[(x_train, y_train), (x_test, y_test)])

xgb_roc_score = roc_auc_score(y_test, xgb_clf.predict_proba(x_test)[:, 1], average='macro')
print('ROC AUC:{0:.4f}'.format(xgb_roc_score))

from sklearn.model_selection import GridSearchCV
params = {'max_depth':[5, 6], 'min_child_weight':[1, 3], 'colsample_bytree':[0.5, 0.75]}

# cv는 3으로 지정
gridcv = GridSearchCV(xgb_clf, param_grid=params, cv=3)
gridcv.fit(x_train, y_train, early_stopping_rounds=30, eval_metric="auc", eval_set=[(x_train, y_train), (x_test, y_test)])

print('GridSearchCV 최적 파라미터:', gridcv.best_params_)

xgb_roc_score = roc_auc_score(y_test, gridcv.predict_proba(x_test)[:, 1], average='macro')
print('ROC AUC: {0:.4F}'.format(xgb_roc_score))


from xgboost import plot_importance
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(10, 8))
plot_importance(xgb_clf, ax=ax, max_num_features=20, height=.4)
plt.show()

'''