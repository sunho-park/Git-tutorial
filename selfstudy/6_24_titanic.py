import pandas as pd
titanic_df = pd.read_csv(r'C:\Users\bitcamp\Desktop\titanic\train.csv')
print(titanic_df.head(3))
print(titanic_df.shape)
print(titanic_df.info())
print(titanic_df.describe())

value_counts = titanic_df['Pclass'].value_counts()
print(value_counts)
# value_counts2=titanic_df['Age'].value_counts()
# print(value_counts2)
print(type(titanic_df))

titanic_pclass = titanic_df['Pclass']
print(type(titanic_pclass))
print(titanic_pclass.head())

titanic_df['Age_0']=0
print(titanic_df.head(3))

# add new columns
titanic_df['Age_by_10'] = titanic_df['Age']*10
titanic_df['Family_No'] = titanic_df['SibSp'] + titanic_df['Parch']+1
print(titanic_df.head(3))

# update columns
titanic_df['Age_by_10'] = titanic_df['Age_by_10']+100
print(titanic_df.head(3))

titanic_drop_df = titanic_df.drop('Age_0', axis=1)
print(titanic_drop_df.head(3))

# inplace 
print('# inplace')
# titanic_df = titanic_df.drop(['Age_0', 'Age_by_10', 'Family_No'], axis=1, inplace=True)
drop_result = titanic_df.drop(['Age_0', 'Age_by_10', 'Family_No'], axis=1, inplace=True)
print('inplace=True 로 drop 후 반환된 값', drop_result) # None 값이 반환되므로 titanic_df = titanic_df.drop 으로 하면 안됨
print(titanic_df.head(3))

# row 삭제
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 15)
print('#### before axis 0 drop ####')
print(titanic_df.head(3))

titanic_df.drop([0, 1, 2], axis=0, inplace=True)
print('#### after axis 0 drop ####')
print(titanic_df.head(3))




