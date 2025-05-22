import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# 加载红酒和白酒数据
red_wine = pd.read_csv(r"e:\New_Project(python)\数据挖掘\data\winequality-red.csv", sep=';')
white_wine = pd.read_csv(r"e:\New_Project(python)\数据挖掘\data\winequality-white.csv", sep=';')

# 添加类型标签：红酒0，白酒1
red_wine['type'] = 0
white_wine['type'] = 1

# 合并数据
wine = pd.concat([red_wine, white_wine], axis=0, ignore_index=True)

# 取特征和标签
X = wine.drop(columns=['quality', 'type']).values
y = wine['type'].values

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# PCA
# 1. 计算协方差矩阵
cov_matrix = np.cov(X_scaled, rowvar=False)

# 2. 求协方差矩阵的特征值和特征向量
eig_vals, eig_vecs = np.linalg.eigh(cov_matrix)

# 3. 对特征值排序，选取最大的前2个
sorted_indices = np.argsort(eig_vals)[::-1]
top2_indices = sorted_indices[:2]
top2_eig_vecs = eig_vecs[:, top2_indices]

# 4. 投影到主成分空间
X_pca = np.dot(X_scaled, top2_eig_vecs)


# 可视化：按颜色区分红酒与白酒
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[y==0, 0], X_pca[y==0, 1], alpha=0.5, s=10, c='red', label='Red Wine')
plt.scatter(X_pca[y==1, 0], X_pca[y==1, 1], alpha=0.5, s=10, c='gold', label='White Wine')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA of Wine Quality')
plt.legend()
plt.grid(True)
plt.show()

# 可选：保存降维结果
pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
pca_df['type'] = y
pca_df.to_csv(r"e:\New_Project(python)\数据挖掘\data\wine_pca2d.csv", index=False)