import numpy as np
import matplotlib.pyplot as plt

def competitive_kmeans(data, K=10, epoch=50, a=0.2, r=1): # default a=0.2, r=0.5
    # 初始化簇中心（随机选择K个数据点作为初始簇中心）
    centers = data[np.random.choice(data.shape[0], K, replace=False)]

    for _ in range(epoch):
        # 分配阶段：将每个点分配给最近的簇中心
        distances = np.linalg.norm(data[:, np.newaxis] - centers, axis=2)
        labels = np.argmin(distances, axis=1)

        # 记录当前的簇中心以检查是否收敛
        old_centers = centers.copy()

        # 更新阶段：重新计算每个簇的中心，并进行惩罚更新
        new_centers = []
        for i in range(K):
            # 获取当前簇和属于该簇的点
            cluster_points = data[labels == i]
            if len(cluster_points) == 0:
                continue  # 如果某个簇没有数据点，跳过

            # 计算簇内点的均值
            cluster_mean = cluster_points.mean(axis=0)

            # 计算第二近的点集并应用惩罚
            second_nearest_points = data[labels != i]
            second_distances = distances[labels != i, i]  # 到第i个簇中心的距离
            if len(second_distances) > 0:
                # 获取到该簇的距离最小的点
                second_nearest_points = second_nearest_points[np.argsort(second_distances)[:len(cluster_points)]]
                second_mean = second_nearest_points.mean(axis=0)
                cluster_mean = cluster_mean - a * second_mean

            new_centers.append(cluster_mean)

        centers = np.array(new_centers)

        # 移除点数小于阈值的簇（每轮限制至多1个簇被移除）
        valid_clusters = []
        cluster_sizes = np.array([np.sum(labels == i) for i in range(len(centers))])
        # 找到点数最少的簇
        min_size_cluster = np.argmin(cluster_sizes)
        # 检查该簇是否低于移除阈值
        if cluster_sizes[min_size_cluster] < r * len(data) / K:
            # 仅移除该点数最少的簇
            valid_clusters = [centers[i] for i in range(len(centers)) if i != min_size_cluster]
        else:
            # 如果没有簇低于阈值，则保持原有簇
            valid_clusters = centers
        
        # 检查是否有簇被删除
        if len(valid_clusters) != len(centers):
            centers = np.array(valid_clusters)
            K = len(centers)  # 更新簇数
            continue  # 跳过收敛检查，进入下一次迭代

        # 收敛检查：如果中心不再更新则提前终止
        if np.allclose(centers, old_centers):
            print("Converged! Epoch:", _+1)
            break

    # 最终计算每个簇的中心
    new_centers = []
    for i in range(K):
        # 获取当前簇和属于该簇的点
        cluster_points = data[labels == i]
        cluster_mean = cluster_points.mean(axis=0)
        new_centers.append(cluster_mean)
    centers = np.array(new_centers)    

    return centers, labels

# 生成数据
np.random.seed(42)
num_points = [50, 100, 150]
cluster_stds = [0.3, 0.5, 0.8]
cluster_1 = np.random.randn(num_points[0], 2) * cluster_stds[0] + np.array([1, 1])
cluster_2 = np.random.randn(num_points[1], 2) * cluster_stds[1] + np.array([5, 5])
cluster_3 = np.random.randn(num_points[2], 2) * cluster_stds[2] + np.array([9, 1])
data = np.vstack([cluster_1, cluster_2, cluster_3])

# 使用竞争学习K-means算法进行聚类
centers, labels = competitive_kmeans(data)

# 可视化结果
plt.figure(figsize=(8, 6))
for i in range(len(centers)):
    cluster_points = data[labels == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i+1}', alpha=0.6)
plt.scatter(centers[:, 0], centers[:, 1], color='black', marker='x', s=100, label='Centers')
plt.legend()
plt.title('Competitive K-means Clustering (a=0.2, r=0.5)')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()
