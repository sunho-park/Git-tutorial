import pandas as pd
titanic_df = pd.read_csv(r'C:\Users\bitcamp\Desktop\titanic\train.csv')

titanic_df['Child_Adult']=titanic_df['Age'].apply(lambda x : 'Child' if x <= 15 else 'Adult')  
print(titanic_df[['Age', 'Child_Adult']].head(8))

titanic_df['Age_cat'] = titanic_df['Age'].apply(lambda x : 'Child' if x<=15 else('Adult' if x <= 60 else 'Elderly'))

print(titanic_df['Age_cat'].value_counts())

# 나이에 따라 세분화된 분류를 수행하는 함수 생성.

def get_category(age):
    cat=''
    if age<=5: cat='Baby'
    elif age <= 12: cat = 'Child'
    elif age <= 18: cat = 'Teenager'
    elif age <= 25: cat = 'Student'
    elif age <= 35: cat = 'Young Adult'
    elif age <= 60: cat = 'Adult'
    else : cat = 'Elderly'

    return cat

# get_category(x) 입력값으로 'Age' 칼럼 값을 받아서 해당하는 cat 반환
titanic_df['Age_cat']= titanic_df['Age'].apply(lambda x : get_category(x))
print(titanic_df[['Age', 'Age_cat']].head())
