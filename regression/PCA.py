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

# 3. 对特征值排序，选取最大的前3个
sorted_indices = np.argsort(eig_vals)[::-1]
top3_indices = sorted_indices[:3]
top3_eig_vecs = eig_vecs[:, top3_indices]

# 4. 投影到主成分空间
X_pca = np.dot(X_scaled, top3_eig_vecs)

# 5. 输出前三个主成分的方差占比
explained_variance_ratio = eig_vals[top3_indices] / np.sum(eig_vals)
print("Explained variance ratio of top 3 principal components:", explained_variance_ratio)
print("Sum of explained variance ratio (top 3):", explained_variance_ratio.sum())

# 6. 可视化三维散点图
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_pca[y==0, 0], X_pca[y==0, 1], X_pca[y==0, 2], alpha=0.5, s=10, c='red', label='Red Wine')
ax.scatter(X_pca[y==1, 0], X_pca[y==1, 1], X_pca[y==1, 2], alpha=0.5, s=10, c='gold', label='White Wine')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_title('3D PCA of Wine Quality')
ax.legend()
plt.show()

# 保存降维结果
pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
pca_df['type'] = y
pca_df.to_csv(r"e:\New_Project(python)\数据挖掘\data\wine_pca2d.csv", index=False)