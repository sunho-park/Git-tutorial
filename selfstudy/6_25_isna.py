import pandas as pd
titanic_df = pd.read_csv(r'C:\Users\bitcamp\Desktop\titanic\train.csv')
print(titanic_df.isna().head())

print(titanic_df.isna().sum())

titanic_df['Cabin'] = titanic_df['Cabin'].fillna('C000')
# titanic_df['Cabin'].fillna('C000', inplace=True)
print(titanic_df)

titanic_df['Age'] = titanic_df['Age'].fillna(titanic_df['Age'].mean())
titanic_df['Embarked'] = titanic_df['Embarked'].fillna('S')
titanic_df.isna().sum()
