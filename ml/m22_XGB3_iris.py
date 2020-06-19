# 과적함 방지
# 1. 훈련 데이터량을 늘린다.
# 2. 피처수를 줄인다.
# 3. regularization
from sklearn.datasets import load_iris
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRFClassifier, plot_importance

# 회귀 모델
iris = datasets.load_iris()

x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,
                                    shuffle = True, random_state = 66)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# 이 정도만 조작해 주면 됨
n_estimators = 1000           # The number of trees in the forest.
learning_rate = 1             # 학습률
colsample_bytree = None       # 트리의 샘플 / 실전0.6 ~ 0.9 사이 씀 / 실전 1씀
colsample_bylevel  = 0.9      # [기본설정값: 1]: subsample, colsample_bytree 두 초모수 설정을 통해서 이미 의사결정나무 모형 개발에 사용될 변수갯수와 관측점 갯수를 사용했는데 추가로 colsample_bylevel을 지정하는 것이 특별한 의미를 갖는지 의문이 듦.
max_depth = 29                # [기본설정값: 6]: 과적합 방지를 위해서 사용되는데 역시 CV를 사용해서 적절한 값이 제시되어야 하고 보통 3-10 사이 값이 적용된다.
n_jobs = -1

# CV 써라
# XGB 속도가 굉장히 빠름, 전처리 결측치 제거 안해줘도 됨

model = XGBRFClassifier(max_depth=max_depth, learning_rate=learning_rate, 
                        n_estimators=n_estimators, 
                        colsample_bylevel=colsample_bylevel,
                        colsample_bytree = colsample_bytree)

model.fit(x_train, y_train)

score = model.score(x_test, y_test) # score는 evaluate
print('점수 :', score)

# print(model.feature_importances_)
plot_importance(model)
# plt.show()

# XGBRFClassifier 점수 : 0.9666666666666667

# XGBClassifier 점수 : 0.8666666666666667



