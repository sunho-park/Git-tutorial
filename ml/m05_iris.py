from sklearn.datasets import load_iris
from sklearn import  datasets
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import  train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import MinMaxScaler, StandardScaler, robust_scale, normalize


iris = datasets.load_iris()
x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 모델
# model = KNeighborsClassifier()
# acc :  0.9466666666666667

model = RandomForestClassifier()
# score :  0.9866666666666667
# acc :  0.9333333333333333

# model = KNeighborsRegressor()
# error

# model = RandomForestRegressor()
# error

# model = LinearSVC()
# score :  0.98
# acc :  0.9666666666666667

# model = SVC()
# score :  0.9666666666666667
# acc :  0.9333333333333333


# 훈련
model.fit(x_train, y_train)

# 평가 예측
score = model.score(x_test, y_test)
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
R2 = r2_score(y_test, y_predict)


print('score : ', score)
print("acc : ", acc)
print("test R2 : ", R2) 