import numpy as np
org_array = ([3, 1, 9, 5])
print('원본 행렬: ', org_array)

# np.sort() 로 정렬
sort_array1 = np.sort(org_array)
print('np.sort() 호출 후 반환된 정렬 행렬', sort_array1)
print('np.sort() 호출 후 반환된 원본 행렬', org_array)

# ndarray.sort() 로 정렬
sort_array2 =org_array.sort()
print('org_array.sort() 호출 후 반환된 행렬', sort_array2)  # None
print('org_array.sort() 호출 후 원본 행렬', org_array)      # [1, 3, 5, 9]

# 내림차순 정렬
sort1_array1_desc = np.sort(org_array)[::-1]
print('내림차순으로 정렬', sort1_array1_desc)

# 행렬이 2차원일 경우
array2d = np.array([[8, 12], [7, 1]])

# argsort

name_array = np.array(['john', 'mike', 'sarah', 'kate', 'samuel'])
score_array = np.array([78, 95, 84, 98, 88])

sort_indices_asc=np.argsort(score_array)

print('성적 오름차순 정렬 시 score_array의 인덱스 : ', sort_indices_asc)
print('성적 오름차순으로 name_array의 이름 출력: ', name_array[sort_indices_asc])

# 행렬 내적
A = np.array([[1, 2, 3], [4, 5, 6]])
B = np.array([[7, 8], [9, 10], [11, 12]])
dot_product = np.dot(A, B)
print('행렬 내적 결과: \n ', dot_product)

# transpose 전치행렬

 
