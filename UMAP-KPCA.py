import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import umap.umap_ as umap
from sklearn.decomposition import KernelPCA
from sklearn import decomposition
import matplotlib.pyplot as plt

data = pd.read_csv('train.csv')
print(data.shape)
data=pd.get_dummies(data)
print(data.shape)
data.drop(['id'], axis=1)

um = umap.UMAP(n_neighbors=5, min_dist=0.3, n_components=100)
um_result = um.fit_transform(data)
print(um_result.shape)

kpca = KernelPCA(n_components=20, kernel='rbf', gamma=20)
kpca = kpca.fit(um_result)
kpca_result = kpca.transform(um_result)
print(kpca_result.shape)

pca = decomposition.PCA(n_components=2)
pca.fit(kpca_result)
pca_result = pca.transform(kpca_result)
print(pca_result.shape)

plt.figure(figsize=(12, 8))
plt.title('Factorize')
plt.scatter(pca_result[:, 0], pca_result[:, 1])
