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
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import accuracy_score, r2_score

# dataset = load_boston()
# x = dataset.data
# y = dataset.target

x, y = load_boston(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

model = XGBRegressor(n_estimators=100, learning_rate=0.1)  # 나무의 갯수(n_estimators)는 epoch 

model.fit(x_train, y_train, verbose=True, eval_metric=["logloss", "rmse"], eval_set =[(x_train, y_train), (x_test, y_test)], early_stopping_rounds=100)
# eval_set은 validation_0이 x_train, y_train// validation1이 x_test, y_test
# {'validation_0': {'rmse': [21.584942, 19.552324, 17.718475]} , 'validation_1': {'rmse': [21.684599, 19.621567, 17.763321]}}

# train test val val지표가 중요

# rmse, mae, logloss, error(설명 error가 accuracy), auc(설명 accuracy친구)

results = model.evals_result() # XGB 에서 사용
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
ax.plot(x_axis, results['validation_0']['rmse'], label='Train')
ax.plot(x_axis, results['validation_1']['rmse'], label='Test')
ax.legend()
plt.ylabel('Rmse')
plt.title('XGBoost Rmse')
plt.show()

