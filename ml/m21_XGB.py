# m17 copy
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from xgboost import XGBClassifier

# xgb 사이킨런에서 제공하지 않음

cancer = load_breast_cancer()

x_train, x_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, test_size=.2, random_state=42
)

# model = DecisionTreeClassifier(max_depth=3)
# model = RandomForestClassifier() 
# model = GradientBoostingClassifier()
model = XGBClassifier()

# max_features : 기본값써라
# n_estimators : 클수록 좋다!, 단점 메모리 짱 차지, 기본값 100 
# n_jobs = -1 :  병렬처리
 

model.fit(x_train, y_train)

acc = model.score(x_test, y_test)

print(model.feature_importances_)

import matplotlib.pyplot as plt
import numpy as np

print(cancer.data.shape)
print('cancer.data.shape[1] : ', cancer.data.shape[1])  # 30

def plot_feature_importances_cancer(model):
    n_features = cancer.data.shape[1]  # 30
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')  #


    plt.yticks(np.arange(n_features), cancer.feature_names)     # 축에 구간 설정, 이름 표시
    plt.xlabel("Feature Importances")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)                                    # 축의 제한

plot_feature_importances_cancer(model)
plt.show()

# 트리구조의 장점 : 전처리가 필요없다. 단점 : 과적합잘됨

print("acc : ", acc)