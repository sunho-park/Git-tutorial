'''
m29_eval1_SFM.py
m29_eval2_SFM.py
m29_eval3_SFM.py 에 save를 적용하시요.


save 이름에는 평가지표를 첨가해서
가장 좋은 SFM용 save 파일을 나오도록 할 것.
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
print("acc : ", score)
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

from joblib import dump, load
dump(model, "./model/xgb_save/iris_acc.model")
print("저장됬다.")

model2 = load("./model/xgb_save/iris_acc.model")
print("불러왔다.")

acc = model.score(x_test, y_test)
print("acc : ", acc)

y_pred = model2.predict(x_test)

r2 = r2_score(y_test, y_pred)
print("r2 : ", r2)