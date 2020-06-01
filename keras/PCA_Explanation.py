import numpy as np
from sklearn.decomposition import PCA

x = np.array([[-1, -1, 3], [-2, -1, 3], [-3, -2, -1], [1, 1, -1], [2, 1, 1], [3, 2, 1]])
pca = PCA(n_components=1)
pca.fit(x)
pca.transform(x)

print('x: \n', x)
print('pca.transform(x) : \n', pca.transform(x))
# print('pca.transform(x) : \n', pca.transform(x))


#  6행 3열 인데 pca 하면 값이 압축된 값이 나옴
# 숫자가 어떻게 나왔는지 알필요없고 
# 분산된 데이터들이 기울기 선쪽으로 압축됨
# PCA하기전 표준화해야함