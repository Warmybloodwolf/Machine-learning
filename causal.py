# jdk-results.csv

import pandas as pd
import numpy as np
from causallearn.search.ConstraintBased.PC import pc
from causallearn.graph.GeneralGraph import GeneralGraph
import matplotlib.pyplot as plt
import networkx as nx

# 1. 加载数据集
data_file = "jdk-results.csv"  # 数据集的文件名
data = pd.read_csv(data_file)

# 2. 选择需要分析的变量列
numeric_columns = [
    "source_ncss", "test_classes", "test_functions", "test_ncss",
    "test_javadocs", "debug", "obfuscate", "optimize", "parallelgc",
    "num_bytecode_ops", "total_unit_test_time", "allocated_bytes",
    "jar_file_size_bytes", "compile_time_ms"
]
data_numeric = data[numeric_columns]

# 转换为 numpy 数组
data_array = data_numeric.to_numpy()

# 验证数据是否正确过滤
print("输入数据的列名：", data_numeric.columns.tolist())
print("输入数据的形状：", data_array.shape)

# 3. 使用 PC 算法进行因果发现
graph = pc(data_array, alpha=0.1)  # 设置显著性水平

# 4. 手动将 GeneralGraph 转换为 NetworkX 图
causal_graph = nx.DiGraph()  # 创建有向图

# 添加节点
for i, column in enumerate(numeric_columns):
    causal_graph.add_node(i, label=column)

edges = graph.G.get_graph_edges()  # 获取 GeneralGraph 的边

# 添加因果边
for edge in edges:
    node1 = edge.node1.get_name()  # 获取节点名称
    node2 = edge.node2.get_name()  # 获取节点名称
    
    # 转换为整数索引（去掉 'X'）
    node1_index = int(node1[1:])-1
    node2_index = int(node2[1:])-1

    print(f"添加因果边：{numeric_columns[node1_index]} -> {numeric_columns[node2_index]}")
    causal_graph.add_edge(node1_index, node2_index)  # 添加边

# 添加节点标签
labels = {i: numeric_columns[i] for i in range(len(numeric_columns))}

# 5. 使用 NetworkX 和 Matplotlib 绘制因果图
plt.figure(figsize=(9, 9))  

# 绘制图形
nx.draw(
    causal_graph,
    pos = nx.circular_layout(causal_graph),
    with_labels=True,
    labels=labels,
    node_color="skyblue",
    font_size=12,           
    node_size=2000,        
    font_weight="bold",
    arrowsize=16,          
    edge_color="gray",     
    alpha=1              
)

# 设置标题
plt.title("Causal Graph", fontsize=16)
plt.show()
