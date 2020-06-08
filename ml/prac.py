import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
import warnings



warnings.filterwarnings('ignore')


iris = pd.read_csv('./data/csv/iris.csv', header=0)

x = iris.iloc[:, 0:4]
y = iris.iloc[:, 4]


kfold = KFold(n_splits=5, shuffle=True)

allAlgorithms = all_estimators(type_filter='classifier')

for (name, algorithm) in allAlgorithms:
    model = algorithm()

    scores = cross_val_score(model, x, y, cv=kfold) # cv = cross validation

    print(name, "의 정답률 = ")
    print(scores)

import sklearn
print(sklearn.__version__)


