# 원본 파일 다시 로딩
import pandas as pd
titanic_df = pd.read_csv(r'C:\Users\bitcamp\Desktop\titanic\train.csv')

# Index 객체 추출
indexes = titanic_df.index
print(indexes)
# Index 객체를 실제 값 array 로 변환
# print('Index 객체 array 값: \n', indexes.values)

series_fair = titanic_df['Fare']
print('Fair Series max 값:', series_fair.max())
print('Fair Series sum 값:', series_fair.sum())
print('sum() Fair Series:', sum(series_fair))
print('Fair Series + 3:\n', (series_fair+3).head(3))

# new index
titanic_reset_df = titanic_df.reset_index(inplace=False)
print(titanic_reset_df.head(3))

print('### before reset_index ###')
value_counts = titanic_df['Pclass'].value_counts()
print(value_counts)
print('value_counts 객체 변수 타입', type(value_counts))
new_value_counts = value_counts.reset_index(inplace=False)
print('### After reset_index ###')
print(new_value_counts)
print('new_value_counts 객체 변수 타입:', type(new_value_counts))

'''
titanic_boolean = titanic_df[titanic_df['Age']>60]
print(type(titanic_boolean))
print(titanic_boolean)
'''

titanic_boolean = titanic_df[titanic_df['Age']>60][['Name', 'Age']]
print(type(titanic_boolean))
print(titanic_boolean)

print(titanic_df[(titanic_df['Age']>60)& (titanic_df['Pclass']==1) & (titanic_df['Sex']=='female')])
# or
cond1 = titanic_df['Age']>60
cond2 = titanic_df['Pclass']==1
cond3 = titanic_df['Sex']=='female'
print(titanic_df[cond1&cond2&cond3])

# 정렬
titanic_sorted = titanic_df.sort_values(by=['Name'])
print(titanic_sorted.head(3))

titanic_sorted = titanic_df.sort_values(by=['Pclass', 'Name'], ascending=False)
print(titanic_sorted.head(3))

# Aggregation
# Groupby
# titanic_groupby = titanic_df.groupby('Pclass').count()
# print(titanic_groupby)

# titanic_groupby = titanic_df.groupby('Pclass')[['PassengerId', 'Survived']].count()
# print(titanic_groupby)

# print(titanic_df.groupby('Pclass')['Age'].agg([max, min]))

agg_format = {'Age':'max', 'SipSP':'sum', 'Fare':'mean'}
titanic_df.groupby('Pclass').agg(agg_format)
# print(titanic_df)
print(titanic_df.groupby('Pclass').agg(agg_format))

