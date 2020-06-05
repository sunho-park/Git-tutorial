import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 와인 데이터 읽기
wine = pd.read_csv('./data/csv/winequality-white.csv', sep=';', header=0)

y = wine['quality']
x = wine.drop('quality', axis=1)

print(x.shape)
print(y.shape)


# y 레이블 축소
newlist = []

for i in list(y):
    if i <=4:
        newlist +=[0]
    elif i <=7:
        newlist +=[1]
    else:
        newlist +=[2]

y=newlist

# 데이터셋 자르기
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# 모델 구성
model=RandomForestClassifier()

# 훈련
model.fit(x_train, y_train)

# 평가
acc = model.score(x_test, y_test)

y_pred = model.predict(x_test)

print("정답률  : ", accuracy_score(y_test, y_pred))
print("acc    : ", acc)