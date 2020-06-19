from sklearn.datasets import load_breast_cancer
from sklearn import  datasets
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import  train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler


breast_cancer = datasets.load_breast_cancer()

x = breast_cancer.data
y = breast_cancer.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.5)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# 모델
# model = KNeighborsClassifier()
# score :  0.9314586994727593
# acc :  0.9228070175438596

model = RandomForestClassifier()
# score :  0.9753954305799648
# acc :  0.9508771929824561

# model = KNeighborsRegressor()
# ValueError: Classification metrics can't handle a mix of binary and continuous targets

# model = RandomForestRegressor()
# ValueError: Classification metrics can't handle a mix of binary and continuous targets

# model = LinearSVC()
# score :  0.9173989455184535
# acc :  0.9192982456140351

# model = SVC()
# score :  0.9103690685413005
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
print("R2 : ", R2) 