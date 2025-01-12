import numpy as np
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score

# 读取 LIBSVM 格式数据的函数
def read_libsvm_file(file_path, max_index=None):
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
    dense_features = np.zeros((len(features), max_index))
    for i, feature_dict in enumerate(features):
        for index, value in feature_dict.items():
            dense_features[i, index - 1] = value  # 转换为 0 索引基

    return np.array(labels), dense_features

# 文件路径
train_file_path = 'a1a.txt'  
test_file_path = 'a1a.t'    
train_file_path = 'splice.txt'  
test_file_path = 'splice.t'    

# 读取训练集数据，获取最大特征索引
y_train, X_train = read_libsvm_file(train_file_path, max_index=125)

# 使用相同的 max_index 读取测试集数据
y_test, X_test = read_libsvm_file(test_file_path, max_index=125)

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

# # 定义 MLP 分类器
# mlp_classifier = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)

# # 模型训练
# mlp_classifier.fit(X_train, y_train)

# # 模型预测
# y_pred_mlp = mlp_classifier.predict(X_test)

# # 评估 MLP 模型
# print("\nMLP Classification Report:")
# print(classification_report(y_test, y_pred_mlp))
# print("MLP Accuracy:", accuracy_score(y_test, y_pred_mlp))

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


