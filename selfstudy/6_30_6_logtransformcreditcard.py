import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')
# get_clf_eval()
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
    print('정확도 : {0:.4f}, 정밀도:{1:.4f}, 재현율: {2:.4f}, F1:{3:.4f}, ROC-AUC:{4:.4f}'.format(accuracy, precision, recall, f1, roc_auc))


card_df = pd.read_csv('./selfstudy/creditcard.csv')
print(card_df.head())
print(card_df.info())

from sklearn.model_selection import train_test_split


# 인자로 입력받은 DataFrame을 복사한 뒤 Time 칼럼만 삭제하고 복사된 DataFrame 반환
#
'''사이킷런의 StandardScaler 를 이용해 정규 분포 형태로 Amount 피처값 변환하는 로직으로 수정 == 추가''' 

def get_preprocessed_df(df=None):
    df_copy = df.copy()
    # 넘파이의 log1p() 를 이용해 Amount를 로그 변환
    amount_n = np.log1p(df_copy['Amount'])
    df_copy.insert(0, 'Amount_Scaled', amount_n)
    df_copy.drop(['Time', 'Amount'], axis=1, inplace=True)
    return df_copy

# 사전 데이터 가공 후 학습과 테스트 데이터 세트를 반환하는 함수
def get_train_test_dataset(df=None):
    # 인자로 입력된 DataFrame의 사전 데이터 가공이 완료된 복사 DataFrame 반환
    df_copy = get_preprocessed_df(df)
    # DataFrame의 맨 마지막 칼럼이 레이블, 나머지는 피처들
    x_feature = df_copy.iloc[:, :-1]
    y_target = df_copy.iloc[:, -1]
    # train_test_split()으로 학습과 테스트 데이터 분할. stratify=y_target으로 Stratified 기반 분할
    x_train, x_test, y_train, y_test = train_test_split(x_feature, y_target, test_size=.3, random_state=0, stratify=y_target)
    # 학습과 테스트 데이터 세트 반환
    return x_train, x_test, y_train, y_test

# 인자로 사이킷런의 Estimator객체와 학습/테스트 데이터 세트를 입력 받아서 학습/예측/평가 수행.
def get_model_train_eval(model, ftr_train=None, ftr_test=None, tgt_train=None, tgt_test=None):
    model.fit(ftr_train, tgt_train)
    pred = model.predict(ftr_test)
    pred_proba = model.predict_proba(ftr_test)[:, 1]
    get_clf_eval(tgt_test, pred, pred_proba)
   
x_train, x_test, y_train, y_test = get_train_test_dataset(card_df)

# 로지스틱 회귀 
from sklearn.linear_model import LogisticRegression
print('### 로지스틱 회귀 예측 성능 ###')
lr_clf = LogisticRegression()
get_model_train_eval(lr_clf, ftr_train=x_train, ftr_test=x_test, tgt_train=y_train, tgt_test=y_test)

print('### LightGBM 예측 성능###')
lgbm_clf = LGBMClassifier(n_estimators=1000, num_leaves=64, n_jobs=-1)
get_model_train_eval(lgbm_clf, ftr_train=x_train, ftr_test=x_test, tgt_train=y_train, tgt_test=y_test)

# 약간의 성능 개선
 