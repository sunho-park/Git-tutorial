from sklearn.feature_selection import SelectFromModel
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# dataset = load_boston()
# x = dataset.data
# y = dataset.target

x, y = load_boston(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

model = XGBRegressor()
model.fit(x_train, y_train)
score = model.score(x_test, y_test) 
print("R2 : ", score)

# feature engineering
thresholds = np.sort(model.feature_importances_)
print(thresholds)

# 위 Gridsearch 적용

# ============================================================================================================
# ============================================================================================================

# 아래도 Gridsearch 적용


for thresh in thresholds:   # 컬럼수 만큼 돈다! 빙글 빙글
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
                                               # median
     
    select_x_train = selection.transform(x_train)
    # print(select_x_train.shape)

    selection_model = XGBRegressor()
    selection_model.fit(select_x_train, y_train)

    select_x_test = selection.transform(x_test)
    y_pred = selection_model.predict(select_x_test)

    score = r2_score(y_test, y_pred)
    # print("R2 : ", score)

    print("thresh=%.3f, n=%d, R2: %.2f%%" %(thresh, select_x_train.shape[1], score*100.0))


# median ? 
# 파라미터 컬럼의 중요도
# SelectFromModel 에 그리드 서치까지 엮어라.
# 데이콘 적용해라 71개 컬럼
# 일요일까지 제출 소스 메일로 보낼 것 메일 제목 : 말똥이 10등
# 전처리 결측치 안해도됨 XGB 장점
# 
'''
[0.00134153 0.00363372 0.01203115 0.01220458 0.01447935 0.01479119
 0.0175432  0.03041655 0.04246345 0.0518254  0.06949984 0.30128643
 0.42848358]
median=0.001, n=13, R2: 92.21%
median=0.004, n=12, R2: 92.16%
median=0.012, n=11, R2: 92.03%
median=0.012, n=10, R2: 92.19%
median=0.014, n=9, R2: 93.08%
median=0.015, n=8, R2: 92.37%
median=0.018, n=7, R2: 91.48%
median=0.030, n=6, R2: 92.71%
median=0.042, n=5, R2: 91.74%
median=0.052, n=4, R2: 92.11%
median=0.069, n=3, R2: 92.52%
median=0.301, n=2, R2: 69.41%
median=0.428, n=1, R2: 44.98%
'''