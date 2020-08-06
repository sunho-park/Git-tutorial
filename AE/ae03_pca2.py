import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_diabetes

dataset = load_diabetes()

X = dataset.data
Y = dataset.target

print(X.shape)
print(Y.shape)

# pca = PCA(n_components=8)
# x2 = pca.fit_transform((X))
# pca_evr = pca.explained_variance_ratio_
# print(pca_evr)
# print(sum(pca_evr))

pca = PCA()
pca.fit(X)
cumsum = np.cumsum(pca.explained_variance_ratio_) # 합이 누적되서 계산됨
print(cumsum)

n_components = np.argmax(cumsum >= 0.94) + 1
print(cumsum>=0.94)
print(n_components)