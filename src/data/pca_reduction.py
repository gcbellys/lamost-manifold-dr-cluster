#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import logging
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def perform_pca_reduction(input_file, output_file, variance_ratio=0.95):
    """
    对特征矩阵进行PCA降维
    
    参数:
    input_file: 输入CSV文件路径
    output_file: 输出CSV文件路径
    variance_ratio: 需要保留的方差比例，默认0.95
    """
    # 读取数据
    logger.info(f"正在读取文件: {input_file}")
    try:
        data = pd.read_csv(input_file)
    except Exception as e:
        logger.error(f"读取文件失败: {str(e)}")
        return
    
    # 分离特征和标签
    X = data.drop(['obsid', 'type'], axis=1)
    labels = data[['obsid', 'type']]
    
    # 标准化处理
    logger.info("正在进行标准化处理...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA降维
    logger.info("正在进行PCA降维...")
    pca = PCA(n_components=variance_ratio)
    X_reduced = pca.fit_transform(X_scaled)
    
    # 输出降维信息
    n_components = pca.n_components_
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    
    logger.info(f"原始特征维度: {X.shape[1]}")
    logger.info(f"降维后特征维度: {n_components}")
    logger.info(f"保留的方差比例: {cumulative_variance_ratio[-1]:.4f}")
    
    # 创建降维后的特征列名
    feature_columns = [f'PC{i+1}' for i in range(n_components)]
    
    # 合并降维后的特征和标签
    df_reduced = pd.DataFrame(X_reduced, columns=feature_columns)
    df_reduced = pd.concat([labels, df_reduced], axis=1)
    
    # 保存结果
    df_reduced.to_csv(output_file, index=False)
    logger.info(f"降维后的数据已保存至: {output_file}")
    
    # 保存PCA模型信息
    pca_info = {
        'n_components': n_components,
        'explained_variance_ratio': explained_variance_ratio,
        'cumulative_variance_ratio': cumulative_variance_ratio,
        'feature_names': feature_columns
    }
    
    # 保存PCA信息到CSV
    pca_info_file = str(output_file).replace('.csv', '_pca_info.csv')
    pd.DataFrame({
        'PC': range(1, n_components + 1),
        'Explained_Variance_Ratio': explained_variance_ratio,
        'Cumulative_Variance_Ratio': cumulative_variance_ratio
    }).to_csv(pca_info_file, index=False)
    logger.info(f"PCA信息已保存至: {pca_info_file}")
    
    # 生成详细说明文档
    doc_file = str(output_file).replace('.csv', '_说明文档.txt')
    with open(doc_file, 'w', encoding='utf-8') as f:
        f.write("PCA降维结果说明文档\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("1. 输出文件说明\n")
        f.write("-" * 30 + "\n")
        f.write(f"主文件: {output_file}\n")
        f.write(f"PCA信息文件: {pca_info_file}\n")
        f.write(f"说明文档: {doc_file}\n\n")
        
        f.write("2. 数据格式说明\n")
        f.write("-" * 30 + "\n")
        f.write("2.1 主文件格式 (reduced_features.csv):\n")
        f.write("   - 行: 每个样本（恒星光谱）\n")
        f.write("   - 列: \n")
        f.write("     * obsid: 观测ID\n")
        f.write("     * type: 恒星类型\n")
        f.write(f"     * PC1-PC{n_components}: 主成分特征\n\n")
        
        f.write("2.2 PCA信息文件格式 (reduced_features_pca_info.csv):\n")
        f.write("   - 行: 每个主成分\n")
        f.write("   - 列: \n")
        f.write("     * PC: 主成分编号\n")
        f.write("     * Explained_Variance_Ratio: 该主成分解释的方差比例\n")
        f.write("     * Cumulative_Variance_Ratio: 累积方差解释比例\n\n")
        
        f.write("3. 处理统计信息\n")
        f.write("-" * 30 + "\n")
        f.write(f"原始特征维度: {X.shape[1]}\n")
        f.write(f"降维后特征维度: {n_components}\n")
        f.write(f"保留的方差比例: {cumulative_variance_ratio[-1]:.4f}\n")
        f.write(f"样本数量: {len(data)}\n\n")
        
        f.write("4. 使用说明\n")
        f.write("-" * 30 + "\n")
        f.write("4.1 主文件 (reduced_features.csv):\n")
        f.write("    - 用于后续机器学习任务\n")
        f.write("    - 每行代表一个恒星光谱样本\n")
        f.write("    - 前两列为标识信息，后续列为降维后的特征\n\n")
        
        f.write("4.2 PCA信息文件 (reduced_features_pca_info.csv):\n")
        f.write("    - 用于分析降维效果\n")
        f.write("    - 可以查看每个主成分的重要性\n")
        f.write("    - 帮助确定最优的主成分数量\n")
    
    logger.info(f"说明文档已生成: {doc_file}")

def main():
    # 设置路径
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data'
    processed_dir = data_dir / 'processed'
    log_dir = Path('/home/lamost-manifold-dr-cluster/log')
    
    # 确保日志目录存在
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置输入输出文件路径
    input_file = processed_dir / 'feature_matrix.csv'
    output_file = processed_dir / 'reduced_features.csv'
    
    # 设置日志文件路径
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'pca_reduction_{timestamp}.log'
    
    # 添加文件日志处理器
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # 检查输入文件是否存在
    if not input_file.exists():
        logger.error(f"输入文件 {input_file} 不存在")
        logger.removeHandler(file_handler)
        return
    
    # 执行PCA降维
    perform_pca_reduction(input_file, output_file)
    
    # 移除文件日志处理器
    logger.removeHandler(file_handler)
    logger.info(f"日志已保存至 {log_file}")

if __name__ == "__main__":
    main() 