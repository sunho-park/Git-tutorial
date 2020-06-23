from sklearn.feature_selection import SelectFromModel
import numpy as np
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston, load_breast_cancer
from sklearn.metrics import accuracy_score, r2_score

# x, y = load_boston(return_X_y=True)
x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

# model = XGBRegressor(n_estimators=1000, learning_rate=0.1)  # 나무의 갯수(n_estimators)는 epoch 
model = XGBClassifier(n_estimators=1000, learning_rate=0.1)  # 나무의 갯수(n_estimators)는 epoch 

model.fit(x_train, y_train, verbose=True, eval_metric="error", eval_set =[(x_train, y_train), (x_test, y_test)])

# rmse, mae, logloss, error(설명 error가 accuracy), auc(설명 accuracy친구)

results = model.evals_result()
# print("eval's results : ", results)

# print("r2 Score : %.2f%%:" %(r2*100.0))

y_pred = model.predict(x_test)
acc = accuracy_score(y_pred, y_test)
print("acc : ", acc)

#####################################################################################################
# import pickle   # 파이썬에서 제공한다.

# from joblib import dump, load
import joblib
# pickle.dump(model, open("./model/xgb_save/cancer.pickle.dat", "wb")) # wb형식으로 저장하겠다.
joblib.dump(model, "./model/xgb_save/cancer.joblib.dat")
print("저장됬다.")

# model2 = pickle.load(open("./model/xgb_save/cancer.pickle.dat", "rb"))
model2 = joblib.load("./model/xgb_save/cancer.joblib.dat")
print('불러왔다.')

y_pred = model2.predict(x_test)
acc = accuracy_score(y_pred, y_test)
print("acc : ", acc)

