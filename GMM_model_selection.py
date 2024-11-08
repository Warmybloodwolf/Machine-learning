import itertools
import numpy as np
from sklearn import mixture
import pandas as pd

# 生成表格用的空列表
results = []

# 随机种子以确保结果可复现
np.random.seed(0)

# 随机生成多个数据集
num_datasets = 10
for _ in range(num_datasets):
    # 随机样本尺寸、维度、真实簇数
    n_samples = np.random.randint(100, 1000)
    n_features = np.random.randint(2, 5)
    n_clusters = np.random.randint(2, 8)
    
    # 随机生成数据
    X = []
    for i in range(n_clusters):
        mean = np.random.rand(n_features) * 10
        cov = np.diag(np.random.rand(n_features) * 2)
        X.append(np.random.multivariate_normal(mean, cov, size=n_samples // n_clusters))
    X = np.vstack(X)

    # 1. 传统GMM方法，使用AIC和BIC选择最佳聚类数
    best_aic_k, best_bic_k = None, None
    lowest_aic, lowest_bic = np.infty, np.infty
    
    for k in range(1, 10):  # 测试不同簇数范围
        gmm = mixture.GaussianMixture(n_components=k, covariance_type="full").fit(X)
        aic = gmm.aic(X)
        bic = gmm.bic(X)
        
        if aic < lowest_aic:
            lowest_aic = aic
            best_aic_k = k
        if bic < lowest_bic:
            lowest_bic = bic
            best_bic_k = k
    
    # 2. Dirichlet过程GMM方法确定簇数
    dpgmm = mixture.BayesianGaussianMixture(n_components=10, covariance_type="full").fit(X)
    dpgmm_k = len(np.unique(dpgmm.predict(X)))
    
    # 将结果记录到列表，增加下划线判断
    results.append({
        "样本尺寸": n_samples,
        "维度": n_features,
        "真实簇数": n_clusters,
        "AIC估计的最佳簇数": f"\\underline{{{best_aic_k}}}" if best_aic_k == n_clusters else best_aic_k,
        "BIC估计的最佳簇数": f"\\underline{{{best_bic_k}}}" if best_bic_k == n_clusters else best_bic_k,
        "Dirichlet过程估计的最佳簇数": f"\\underline{{{dpgmm_k}}}" if dpgmm_k == n_clusters else dpgmm_k,
    })

# 将结果转换为DataFrame
df_results = pd.DataFrame(results)

# 将DataFrame转换为LaTeX表格格式，并打印结果
latex_table = df_results.to_latex(index=False, escape=False, column_format="|c|c|c|c|c|c|")
print(latex_table)

