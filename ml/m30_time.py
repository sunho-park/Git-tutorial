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

model = XGBRegressor()
model.fit(x_train, y_train)
score = model.score(x_test, y_test) 
print("R2 : ", score)

# feature engineering
thresholds = np.sort(model.feature_importances_)
print(thresholds)

# time 추가
import time
start = time.time()

for thresh in thresholds:   # 컬럼수 만큼 돈다! 빙글 빙글
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
                                               # median
    select_x_train = selection.transform(x_train)
    # print(select_x_train.shape)

    selection_model = XGBRegressor(n_estimators=1000)
    selection_model.fit(select_x_train, y_train)

    select_x_test = selection.transform(x_test)
    y_pred = selection_model.predict(select_x_test)

    score = r2_score(y_test, y_pred)
    # print("R2 : ", score)

    print("thresh=%.3f, n=%d, R2: %.2f%%" %(thresh, select_x_train.shape[1], score*100.0))


#############################################################################################

start2 = time.time()

for thresh in thresholds:   # 컬럼수 만큼 돈다! 빙글 빙글
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
                                               # median
    select_x_train = selection.transform(x_train)
    # print(select_x_train.shape)

    selection_model = XGBRegressor(n_jobs=6, n_estimators=1000)
    selection_model.fit(select_x_train, y_train)

    select_x_test = selection.transform(x_test)
    y_pred = selection_model.predict(select_x_test)

    score = r2_score(y_test, y_pred)
    # print("R2 : ", score)

    print("thresh=%.3f, n=%d, R2: %.2f%%" %(thresh, select_x_train.shape[1], score*100.0))


end = start2 - start # 추가
print("그냥 걸린 시간 : ", end)

end2 = time.time() - start2 # 추가
print("잡스 걸린 시간 : ", end2)
