# 가장 큰 장점 xgboost에 비해 장점 더 빠른 학습시간, 예측 수행시간, 더 작은 메모리 사용량, 원한잇코딩 사용 X 
# Light GBM 단점 적은 데이터세트에 적용할 경우 과적합 발생하기 쉽다. 10000건 이하의 데이터 세트
# 균형 트리 분할에 비해 리프 중심 트리 분할 예측오류손실 최대화

# LightGBM의 파이썬 패키지인 lightgbm에서 LGBMClassifier 임포트
from lightgbm import LGBMClassifier

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# get_clf_eval
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score, roc_auc_score

def get_clf_eval(y_test, pred=None, pred_proba=None):
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)    
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    # ROC - AUC 추가
    roc_auc = roc_auc_score(y_test, pred_proba)
    print('오차 행렬')
    print(confusion)
    # ROC - AUC print 추가
    print('정확도 : {0:.4f}, 정밀도:{1:.4f}, 재현율: {2:.4f}, F1:{3:.4f}, AUC:{4:.4f}'.format(accuracy, precision, recall, f1, roc_auc))


dataset = load_breast_cancer()
ftr = dataset.data
target = dataset.target

x_train, x_test, y_train, y_test = train_test_split(ftr, target, test_size=.2, random_state=156)

# 앞서 XGBoost 와 동일하게 n_estimator는 400 설정
lgbm_wrapper = LGBMClassifier(n_estimators=400)

# LightGBM 도 XGBoost 와 동일하게 조기 중단 수행 가능.
evals = [(x_test, y_test)]
lgbm_wrapper.fit(x_train, y_train, early_stopping_rounds=100, eval_metric='logloss', eval_set=evals, verbose=True)

preds = lgbm_wrapper.predict(x_test)
pred_proba = lgbm_wrapper.predict_proba(x_test)[:, 1]

print(get_clf_eval(y_test, preds, pred_proba))

# plot_importance()를 이용해 피처 중요도 시각화
from lightgbm import plot_importance
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 12))
plot_importance(lgbm_wrapper, ax=ax)
plt.show()


