'''
m28_eval2 와 3을 만들것


SelectFromModel 에 

1. 회귀          m29_eval1
2. 분류          m29_eval2
3. 다중 분류     m29_eval3

1) eval 에 'loss' 와 다른 지표 1개 더 추가.
2) earlyStopping 적용
3) plot 으로 그릴것.


4. 결과는 주석으로 소스 하단에 표시.

5. m27 ~ 29 까지 완벽 이해할것!
'''

from sklearn.feature_selection import SelectFromModel
import numpy as np
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, r2_score

# dataset = load_iris()
# x = dataset.data
# y = dataset.target

x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

model = XGBClassifier(n_estimators=100, learning_rate=0.1)  # 나무의 갯수(n_estimators)는 epoch 

model.fit(x_train, y_train, verbose=True, eval_metric=["mlogloss", "merror"], eval_set =[(x_train, y_train), (x_test, y_test)], early_stopping_rounds=100)

results = model.evals_result()
# print("eval's results : ", results)

# y_pred = model.predict(x_test)
# r2 = r2_score(y_pred, y_test)
# print("r2 Score : %.2f%%:" %(r2*100.0))
score = model.score(x_test, y_test)
print("r2 : ", score)
#########################################################################################################
# feature engineering
thresholds = np.sort(model.feature_importances_)
print(thresholds)

for thresh in thresholds:
    selection = SelectFromModel(model, threshold=thresh, prefit=True)

    select_x_train = selection.transform(x_train)

    selection_model = XGBClassifier()
    selection_model.fit(select_x_train, y_train)

    select_x_test = selection.transform(x_test)
    y_pred = selection_model.predict(select_x_test)

    score = r2_score(y_test, y_pred)

    print("thresh=%.3f, n=%d, R2: %.2f%%" %(thresh, select_x_train.shape[1], score*100.0))
