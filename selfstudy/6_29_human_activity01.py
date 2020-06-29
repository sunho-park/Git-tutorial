import pandas as pd
import matplotlib as plt

# features.txt 파일에는 피처 이름 index 와 피처명이 공백으로 분리되어 있음
feature_name_df = pd.read_csv('./selfstudy/human_activity/features.txt', sep='\s+', header=None, names=['column_index', 'column_name'])

print(feature_name_df.head())
print(feature_name_df.shape)

# 피처명 index를 제거하고, 피처명만 리스트 객체로 생성한 뒤 샘플로 10개만 추출
feature_name = feature_name_df.iloc[:, 1].values.tolist()
print('전체 피처명에서 10개만 추출:', feature_name[:10])

feature_dup_df = feature_name_df.groupby('column_name').count()
print(feature_dup_df[feature_dup_df['column_index']>1].count())
feature_dup_df[feature_dup_df['column_index']>1].head()

#### #### ####
def get_new_feature_name_df(old_feature_name_df):
    feature_dup_df = pd.DataFrame(data=old_feature_name_df.groupby('column_name').cumcount(), columns=['dup_cnt'])
    feature_dup_df = feature_dup_df.reset_index()
    
    new_feature_name_df = pd.merge(old_feature_name_df.reset_index(), feature_dup_df, how='outer')
    new_feature_Name_df['column_name']=new_feature_name_df[['column_Name', 'dup_cnt']].apply(lambda x : x[0]+'_'+str(x[1]) if x[1] >0  else x[0], axis=1)   

    new_feature_name_df = new_feature_name_df.drop(['index'], axis=1)
    return new_feature_name_df





