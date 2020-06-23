from sklearn.feature_selection import SelectFromModel
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, r2_score
from lightgbm import LGBMClassifier, LGBMRegressor

# dataset = load_breast_cancer()
# x = dataset.data
# y = dataset.target

x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

model = LGBMClassifier(n_estimators=100, learning_rate=0.1)  # 나무의 갯수(n_estimators)는 epoch 

model.fit(x_train, y_train, verbose=True, eval_metric=["logloss", "error"], eval_set =[(x_train, y_train), (x_test, y_test)], early_stopping_rounds=100)
acc = model.score(x_test, y_test)
# results = model.evals_result()
# print("eval's results : ", results)
y_pred = model.predict(x_test)
# r2 = r2_score(y_pred, y_test)
# print("r2 Score : %.2f%%:" %(r2*100.0))

print("acc : ", acc)
# print("r2 : ", r2)
##########################################################################################
# feature engineering
thresholds = np.sort(model.feature_importances_)
print(thresholds)

for thresh in thresholds:
    selection = SelectFromModel(model, threshold=thresh, prefit=True)

    select_x_train = selection.transform(x_train)

    selection_model = LGBMClassifier()
    selection_model.fit(select_x_train, y_train)

    select_x_test = selection.transform(x_test)
    y_pred = selection_model.predict(select_x_test)

    score = r2_score(y_test, y_pred)

    print("thresh=%.3f, n=%d, R2: %2.f%%" %(thresh, select_x_train.shape[1], score*100.0))

import pickle
pickle.dump(model, open("./model/xgb_save/cancer_acc.dat", "wb"))
print("저장됬다")

model2=pickle.load(open("./model/xgb_save/cancer_acc.dat", "rb"))
print("불러왔다.")
