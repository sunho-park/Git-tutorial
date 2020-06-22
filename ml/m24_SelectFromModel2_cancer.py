# breast cancer
from sklearn.feature_selection import SelectFromModel
import numpy as np
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, r2_score

x, y = load_breast_cancer(return_X_y=True)

print(x.shape) # (569, 30)
print(y.shape) # (569, )

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

model = XGBClassifier()
model.fit(x_train, y_train)
score = model.score(x_test, y_test) 
print("R2 : ", score)

# feature engineering
thresholds = np.sort(model.feature_importances_)

print(thresholds)

for thresh in thresholds:   # 컬럼수 만큼 돈다! 빙글 빙글
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
                                               # median
    select_x_train = selection.transform(x_train)
    # print(select_x_train.shape)

    selection_model = XGBClassifier()
    selection_model.fit(select_x_train, y_train)

    select_x_test = selection.transform(x_test)
    y_pred = selection_model.predict(select_x_test)

    score = r2_score(y_test, y_pred)
    # print("R2 : ", score)

    print("Thresh=%.3f, n=%d, R2: %.2f%%" %(thresh, select_x_train.shape[1], score*100.0))

# median
# 파라미터 컬럼의 중요도
# SelectFromModel 에 그리드 서치까지 엮어라.
# 데이콘 적용해라 71개 컬럼
# 일요일까지 제출 소스 메일로 보낼 것 메일 제목 : 말똥이 10등
# 전처리 결측치 안해도됨 XGB 장점
# 
