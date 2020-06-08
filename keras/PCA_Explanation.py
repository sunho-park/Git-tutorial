import numpy as np
from sklearn.decomposition import PCA

x = np.array([[-1, -1, 3], [-2, -1, 3], [-3, -2, -1], [1, 1, -1], [2, 1, 1], [3, 2, 1]])
pca = PCA(n_components=3)

pca.fit(x)
x = pca.transform(x)

# y = pca.transform(x)

print('x: \n', x)
# print('y: \n', y)
print('pca.transform(x) : \n', pca.transform(x)) # pca.transform 은 한번 더 tramsform 을 하는 것임
# print('pca.transform(x) : \n', pca.transform(x))


#  6행 3열 인데 pca 하면 값이 압축된 값이 나옴
# 숫자가 어떻게 나왔는지 알필요없고 
# 분산된 데이터들이 기울기 선쪽으로 압축됨
# PCA하기전 표준화해야함

from sklearn.preprocessing import MinMaxScaler

m = np.array([[-1, -1, 3], [-2, -1, 3], [-3, -2, -1], [1, 1, -1], [2, 1, 1], [3, 2, 1]])

scaler = MinMaxScaler()

scaler.fit(m)
m = scaler.transform(m)

print('m : \n', m)
print('scaler.transform(m) : \n', scaler.transform(m))

# print('z : ', z)