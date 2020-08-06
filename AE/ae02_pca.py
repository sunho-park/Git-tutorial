import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_diabetes

dataset = load_diabetes()

X = dataset.data
Y = dataset.target

print(X.shape)
print(Y.shape)

pca = PCA(n_components=8)
x2 = pca.fit_transform((X))
pca_evr = pca.explained_variance_ratio_
print(pca_evr)
print(sum(pca_evr))

