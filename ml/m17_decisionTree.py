from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()

x_train, x_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, test_size=.2, random_state=42
)

model = DecisionTreeClassifier(max_depth=3) 

model.fit(x_train, y_train)

acc = model.score(x_test, y_test)

print(model.feature_importances_)
print("acc : ", acc)

'''
[0.         0.         0.         0.         0.         0.
 0.         0.70458252 0.         0.         0.01221069 0.
 0.         0.         0.         0.         0.         0.0162341
 0.         0.0189077  0.05329492 0.05959094 0.05247428 0.
 0.00940897 0.00639525 0.         0.06690062 0.         0.        ]
 '''