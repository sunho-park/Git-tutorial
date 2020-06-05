from sklearn.datasets import boston
from sklearn import datasets

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

boston = datasets.load_boston
x = boston.data
y = boston.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

scaler = StandardScaler
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#모델 

model = RandomForestRegressor()

# 훈련
model.fit(x_train, y_train)

score = model.score(x_test, y_test)
y_pred = model.predict(x_test)

acc = accuracy_score(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
