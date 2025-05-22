import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # 初始化权重和偏置
        self.w = np.zeros(n_features)
        self.b = 0

        for i in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y[idx]))
                    self.b -= self.lr * y[idx]

    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)

# 加载红酒和白酒数据集
def load_wine_data(red_filepath, white_filepath):
    # 加载红酒数据并添加类别标签
    red_wine = pd.read_csv(red_filepath, sep=';')
    red_wine['type'] = -1  # 红酒标记为 -1

    # 加载白酒数据并添加类别标签
    white_wine = pd.read_csv(white_filepath, sep=';')
    white_wine['type'] = 1  # 白酒标记为 1

    # 合并数据集
    data = pd.concat([red_wine, white_wine], axis=0)
    return data

# 数据预处理
def preprocess_wine_data(data):
    # 特征和标签
    X = data.iloc[:, :-1].values  # 特征（去掉最后一列 type）
    y = data['type'].values       # 标签（红酒或白酒）

    # 标准化特征
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return train_test_split(X, y, test_size=0.3, random_state=42)

# 主函数
if __name__ == "__main__":
    
    # 设置文件路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    red_filepath = os.path.join(parent_dir, "data", "winequality-red.csv")
    white_filepath = os.path.join(parent_dir, "data", "winequality-white.csv")
    
    # 加载数据
    wine_data = load_wine_data(red_filepath, white_filepath)

    # 数据预处理
    X_train, X_test, y_train, y_test = preprocess_wine_data(wine_data)

    # 训练自定义 SVM 模型
    svm = SVM(learning_rate=0.001, lambda_param=0.01, n_iters=1000)
    svm.fit(X_train, y_train)

    # 预测与评估
    y_pred = svm.predict(X_test)
    print("分类报告：")
    print(classification_report(y_test, y_pred))
    print("准确率：", accuracy_score(y_test, y_pred))