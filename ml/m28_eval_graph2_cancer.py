from sklearn.feature_selection import SelectFromModel
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, r2_score

# dataset = load_breast_cancer()
# x = dataset.data
# y = dataset.target

x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

model = XGBRegressor(n_estimators=100, learning_rate=0.1)  # 나무의 갯수(n_estimators)는 epoch 

model.fit(x_train, y_train, verbose=True, eval_metric=["logloss", "error"], eval_set =[(x_train, y_train), (x_test, y_test)], early_stopping_rounds=100)
# eval_set은 validation_0이 x_train, y_train// validation1이 x_test, y_test

# train test val val지표가 중요


# rmse, mae, logloss, error(설명 error가 accuracy), auc(설명 accuracy친구)

results = model.evals_result()
print("eval's results : ", results)

y_pred = model.predict(x_test)

r2 = r2_score(y_pred, y_test)
# print("r2 Score : %.2f%%:" %(r2*100.0))
print("r2 : ", r2)

import matplotlib.pyplot as plt

epochs = len(results['validation_0']['logloss'])    # epoch의 길이
x_axis = range(0, epochs)

fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
ax.legend()
plt.ylabel('Log Loss')
plt.title('XGBoost Log Loss')
# plt.show()

fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['error'], label='Train')
ax.plot(x_axis, results['validation_1']['error'], label='Test')
ax.legend()
plt.ylabel('Error')
plt.title('XGBoost Error')
plt.show()

