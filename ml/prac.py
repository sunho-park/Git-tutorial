from sklearn.feature_selection import SelectFromModel
import numpy as np
from xgboost import XGBRFRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import accuracy_score, r2_score

x, y = load_boston(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=.8, shuffle=True, random_state=66)

model = XGBRFRegressor(n_estimators=1000, learning_rate=0.1)

model.fit(x_train, y_train, verbose=True, eval_metric=["rmse", "mae"],  eval_set = [(x_train, y_train), (x_test, y_test)], early_stopping_rounds=30)

results = model.evals_result()
print("eval's results : ", results)

y_pred = model.predict(x_test)

r2 = r2_score(y_pred, y_test)
print('r2 : ', r2)

import matplotlib.pyplot as plt

epochs = len(results['validation_0']['rmse'])
x_axis = range(0, epochs)

fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['rmse'], label='Train')
ax.plot(x_axis, results['validation_1']['rmse'], label='Test')
ax.legend()

plt.ylabel('Log Loss')
plt.title('XGBoost Log Loss')
plt.show()