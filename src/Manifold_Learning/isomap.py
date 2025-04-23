import numpy as np
import pandas as pd
from sklearn.manifold import Isomap
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import connected_components
import matplotlib.pyplot as plt

# 假设X_pca和labels已加载，例如：
# X_pca = np.load('X_pca.npy')  # PCA降维后的特征矩阵
# labels = np.load('labels.npy')  # 光谱类型标签，如['A', 'F', 'G']

# 连通性检查
adj_matrix = kneighbors_graph(X_pca, n_neighbors=10, mode='distance')
n_components, component_labels = connected_components(adj_matrix, directed=False)
if n_components > 1:
    print(f"警告：邻域图有{n_components}个连通分量，建议增加k值或调整数据。")

# 应用Isomap降维
isomap = Isomap(n_neighbors=10, n_components=2)
Y = isomap.fit_transform(X_pca)

# 保存结果
df = pd.DataFrame(Y, columns=['dim1', 'dim2'])
df['label'] = labels
df.to_csv('isomap_results.csv', index=False)
print("降维结果已保存至 'isomap_results.csv'")

# 可视化
unique_labels = np.unique(labels)
plt.figure(figsize=(10, 8))
for label in unique_labels:
    idx = labels == label
    plt.scatter(Y[idx, 0], Y[idx, 1], label=label, s=50)
plt.title('Isomap降维结果')
plt.xlabel('维度1')
plt.ylabel('维度2')
plt.legend()
plt.grid(True)
plt.savefig('isomap_scatter.png')
print("散点图已保存至 'isomap_scatter.png'")