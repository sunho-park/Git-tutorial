import numpy as np
import pandas as pd

col_name1 =['col1']
list1 = [1, 2, 3]
print(type(list1)) # <class 'list'>

array1 = np.array(list1)
print(type(array1)) # <class 'numpy.ndarray'>

print('array1 shape:', array1.shape)

# 리스트를 이용하여 DataFrame 생성
df_list1 = pd.DataFrame(list1, columns=col_name1)
print('1차원 리스트로 만든 DataFrame:\n', df_list1)

# 넘파이 ndarray를 이용해 DataFrame 생성.
df_array1 = pd.DataFrame(array1, columns=col_name1) 
print('1차원 ndarray로 만든 DataFrame:\n', df_array1)

# 2차원 DataFrame
# 3개의 칼럼명
col_name2 = ['col1', 'col2', 'col3']
list2 = [[1, 2, 3], [11, 12, 13]]

array2 = np.array(list2)

df_list2=pd.DataFrame(list2, columns=col_name2)

df_array2=pd.DataFrame(array2, columns=col_name2)

print('2차원으로 list로 만든 DataFrame \n', df_list2)
print('2차원으로 nd.array로 만든 DataFrame \n', df_array2)

# dictionary -> dataframe
# key는 문자열 칼럼명으로 매핑, Values는 리스트 형(또는 ndarray) 칼럼 데이터로 매핑
dict = {'col1':[1, 11], 'col2':[2, 22], 'col3':[3, 33]}
df_dict = pd.DataFrame(dict)
print('딕셔너리로 만든 DataFrame: \n', df_dict)

# DataFrame 을 ndarray로 변환

array3 = df_dict.values
print('df_dict.values 타입:', type(array3), 'df_dict.values shape:', array3.shape)
print(array3)

# DataFrame 을  리스트로 변환
list3 = df_dict.values.tolist()
print('df_dict.values.tolist() 타입:', type(list3))
print(list3)

# DataFrame 을  딕셔너리로 변환
dict3 = df_dict.to_dict('list')
print('\n df_dict.to_dict() 타입', type(dict3))
print(dict3)



