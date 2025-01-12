import numpy as np
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# 读取 LIBSVM 格式数据的函数
def read_libsvm_file(file_path):
    labels = []
    features = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            labels.append(int(parts[0]))  # 第一列为标签
            feature_dict = {}
            for item in parts[1:]:
                index, value = item.split(":")
                feature_dict[int(index)] = float(value)
            features.append(feature_dict)

    # 转换成稠密矩阵
    max_index = max(max(f.keys()) for f in features)
    dense_features = np.zeros((len(features), max_index))
    for i, feature_dict in enumerate(features):
        for index, value in feature_dict.items():
            dense_features[i, index - 1] = value  # 转换为 0 索引基

    return np.array(labels), dense_features

# 文件路径
file_path = 'diabetes.txt'  # 替换为你的文件路径

# 读取数据
labels, features = read_libsvm_file(file_path)

# 数据分割为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

# # 定义 SVM 分类器
# svm_classifier = SVC(kernel='linear', C=0.01, random_state=42)

# # 模型训练
# svm_classifier.fit(X_train, y_train)

# # 模型预测
# y_pred = svm_classifier.predict(X_test)

# # 评估模型
# print("Classification Report:")
# print(classification_report(y_test, y_pred))
# print("Accuracy:", accuracy_score(y_test, y_pred))

# 定义 MLP 分类器
mlp_classifier = MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, random_state=42)

# 模型训练
mlp_classifier.fit(X_train, y_train)

# 模型预测
y_pred_mlp = mlp_classifier.predict(X_test)

# 评估 MLP 模型
print("\nMLP Classification Report:(50,)")
print(classification_report(y_test, y_pred_mlp))
print("MLP Accuracy:", accuracy_score(y_test, y_pred_mlp))

# 定义 MLP 分类器
mlp_classifier = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)

# 模型训练
mlp_classifier.fit(X_train, y_train)

# 模型预测
y_pred_mlp = mlp_classifier.predict(X_test)

# 评估 MLP 模型
print("\nMLP Classification Report:(100,)")
print(classification_report(y_test, y_pred_mlp))
print("MLP Accuracy:", accuracy_score(y_test, y_pred_mlp))

# 定义 MLP 分类器
mlp_classifier = MLPClassifier(hidden_layer_sizes=(100,50), max_iter=500, random_state=42)

# 模型训练
mlp_classifier.fit(X_train, y_train)

# 模型预测
y_pred_mlp = mlp_classifier.predict(X_test)

# 评估 MLP 模型
print("\nMLP Classification Report:(100,50)")
print(classification_report(y_test, y_pred_mlp))
print("MLP Accuracy:", accuracy_score(y_test, y_pred_mlp))

# 定义 MLP 分类器
mlp_classifier = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)

# 模型训练
mlp_classifier.fit(X_train, y_train)

# 模型预测
y_pred_mlp = mlp_classifier.predict(X_test)

# 评估 MLP 模型
print("\nMLP Classification Report:(100,200)")
print(classification_report(y_test, y_pred_mlp))
print("MLP Accuracy:", accuracy_score(y_test, y_pred_mlp))

# 定义 MLP 分类器
mlp_classifier = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)

# 模型训练
mlp_classifier.fit(X_train, y_train)

# 模型预测
y_pred_mlp = mlp_classifier.predict(X_test)

# 评估 MLP 模型
print("\nMLP Classification Report:(100,200,100)")
print(classification_report(y_test, y_pred_mlp))
print("MLP Accuracy:", accuracy_score(y_test, y_pred_mlp))
