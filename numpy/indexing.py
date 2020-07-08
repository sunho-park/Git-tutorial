import numpy as np

array1d = np.arange(start=1, stop=10)
print(array1d)

print(array1d>5)

array3d=array1d[array1d>5]
print(array3d)

boolean_indexes = np.array([False, False, False, False, False, True, True, True, True])

# True 에 해당하는 인덱스의 값을 저장
array3 = array1d[boolean_indexes]

print('불린 인덱스로 필터링 결과 : ', array3)
