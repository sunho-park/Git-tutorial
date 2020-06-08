import pandas as pd   

train = pd.read_csv('./data/dacon/comp1/train.csv', header=0, index_col=0, encoding='UTF8')
test = pd.read_csv('./data/dacon/comp1/test.csv', header=0, index_col=0, encoding='UTF8')

submission = pd.read_csv('./data/dacon/comp1/sample_submission.csv', header=0, index_col=0)


print(train.head())

print(train.isnull().sum()[train.isnull().sum().values > 0])
print(test.isnull().sum()[test.isnull().sum().values > 0])

print(train.isnull().sum()[train.isnull().sum().values > 0].index)

test.filter(regex='_src$',axis=1).head().T.plot()