from sklearn.datasets import load_boston
from  sklearn import  datasets
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import  train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn import metrics
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler


# 데이터 전처리
boston = datasets.load_boston()
x = boston.data
y = boston.target

# print(boston)
# print(boston.keys())

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# 모델
# model = KNeighborsRegressor() 
# score :  0.6145326967139015
# test r2 :  0.49517615133092796

model = RandomForestRegressor()
# score :  0.957640390524976
# test r2 :  0.9114488975233372

model = KNeighborsClassifier()
# ValueError: Unknown label type: 'continuous'

# model = RandomForestClassifier()
# ValueError: Unknown label type: 'continuous'

# model = LinearSVC()
# ValueError: Unknown label type: 'continuous'

# model = SVC()
# ValueError: Unknown label type: 'continuous'

# 훈련
model.fit(x_train, y_train)

# 평가 예측
score = model.score(x_test, y_test)
y_predict = model.predict(x_test)
R2 = r2_score(y_test, y_predict)
acc = accuracy_score(y_test, y_predict)

print('score : ', score)
print("test r2 : ", R2) 
print("acc : ", acc)
# 만약에 acc이 나온다면 acc함정 r2값이 acc로 나오는거라서 주의해야함
# continuous is not supported tells me you're trying to do "something" from regression domain on classification domain.

# from sklearn.metrics import r2_score
# r2 = r2_score(y_test, y_predict)
# print("r2 : ", r2)


# y_predict = model.predict(x_train)
# r2 = metrics.r2_score(y_train, y_predict)
# print("train r2 : ", r2)


# acc = accuracy_score(y_test, y_predict)
# print("acc : ", acc)
 
 