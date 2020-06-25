from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

iris = load_iris()

iris_data = iris.data

iris_label = iris.target
print('iris target 값:', iris_label)
print('iris target 명:', iris.target_names)

# 붓꽃 데이터 세트를 자세히 보기 위해 DataFrame 으로 변환
iris_df = pd.DataFrame(data=iris_data, columns=iris.feature_names)
iris_df['label'] = iris.target
print(iris_df.head(3))
print(type(iris_df))

keys = iris_df.keys()
print('붓꽃 데이터 세트의 키들 : ', keys)