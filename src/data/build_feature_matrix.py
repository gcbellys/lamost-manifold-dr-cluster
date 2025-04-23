#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_data_quality(df, flux_columns, ivar_columns):
    """
    检查数据质量
    
    参数:
    df: 数据DataFrame
    flux_columns: 通量列名列表
    ivar_columns: 逆方差列名列表
    
    返回:
    bool: 数据是否有效
    """
    # 检查缺失值
    missing_flux = df[flux_columns].isnull().sum().sum()
    missing_ivar = df[ivar_columns].isnull().sum().sum()
    if missing_flux > 0 or missing_ivar > 0:
        logger.warning(f"发现缺失值: 通量 {missing_flux}个, 逆方差 {missing_ivar}个")
    
    # 检查负通量值
    negative_flux = (df[flux_columns] < 0).sum().sum()
    if negative_flux > 0:
        logger.warning(f"发现负通量值: {negative_flux}个")
    
    # 检查负逆方差值
    negative_ivar = (df[ivar_columns] < 0).sum().sum()
    if negative_ivar > 0:
        logger.warning(f"发现负逆方差值: {negative_ivar}个")
    
    # 检查零逆方差值
    zero_ivar = (df[ivar_columns] == 0).sum().sum()
    if zero_ivar > 0:
        logger.warning(f"发现零逆方差值: {zero_ivar}个")
    
    return True

def build_feature_matrix(input_file, output_file, reduce_dim=False, n_components=100):
    """
    从对齐后的光谱数据构建特征矩阵
    
    参数:
    input_file: 输入CSV文件路径（对齐后的光谱数据）
    output_file: 输出CSV文件路径（特征矩阵）
    reduce_dim: 是否进行PCA降维（默认为False）
    n_components: PCA降维后的维度（如果reduce_dim为True）
    """
    # 加载数据
    logger.info(f"正在加载文件: {input_file}")
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        logger.error(f"加载文件失败: {str(e)}")
        return
    
    # 提取通量和逆方差列
    flux_columns = [col for col in df.columns if col.startswith('flux_')]
    ivar_columns = [col for col in df.columns if col.startswith('ivar_')]
    
    # 检查数据完整性
    if len(flux_columns) != len(ivar_columns):
        logger.error("通量列和逆方差列数量不匹配")
        return
    
    # 检查数据质量
    if not check_data_quality(df, flux_columns, ivar_columns):
        logger.error("数据质量检查未通过")
        return
    
    # 提取通量列
    flux_df = df[flux_columns]
    
    # 转换为numpy数组
    feature_matrix = flux_df.values
    
    # 标准化特征矩阵
    logger.info("正在进行特征标准化...")
    scaler = StandardScaler()
    feature_matrix = scaler.fit_transform(feature_matrix)
    
    # 可选：PCA降维
    if reduce_dim:
        logger.info(f"正在进行PCA降维，目标维度: {n_components}")
        pca = PCA(n_components=n_components)
        feature_matrix = pca.fit_transform(feature_matrix)
        logger.info(f"PCA降维完成，保留了 {pca.explained_variance_ratio_.sum():.2%} 的方差")
    
    # 提取obsid和type
    obsids = df['obsid']
    types = df['type']
    
    # 构建特征矩阵DataFrame
    if reduce_dim:
        columns = [f'PC_{i+1}' for i in range(n_components)]
    else:
        columns = flux_columns
    feature_df = pd.DataFrame(feature_matrix, columns=columns)
    feature_df.insert(0, 'obsid', obsids)
    feature_df.insert(1, 'type', types)
    
    # 保存特征矩阵
    feature_df.to_csv(output_file, index=False)
    logger.info(f"特征矩阵已保存至 {output_file}")
    
    # 输出统计信息
    logger.info("\n=== 特征矩阵统计信息 ===")
    logger.info(f"样本数量: {len(feature_df)}")
    logger.info(f"特征数量: {len(columns)}")
    logger.info(f"特征值范围: [{feature_matrix.min():.2f}, {feature_matrix.max():.2f}]")
    logger.info(f"特征值均值: {feature_matrix.mean():.2f}")
    logger.info(f"特征值标准差: {feature_matrix.std():.2f}")
    
    # 输出光谱类型分布统计
    logger.info("\n=== 光谱类型分布统计 ===")
    type_counts = feature_df['type'].value_counts()
    for star_type, count in type_counts.items():
        percentage = (count / len(feature_df)) * 100
        logger.info(f"{star_type}型星: {count}个 ({percentage:.1f}%)")
    
    # 输出特征值分布统计
    logger.info("\n=== 特征值分布统计 ===")
    logger.info(f"负值数量: {(feature_matrix < 0).sum()}")
    logger.info(f"负值比例: {(feature_matrix < 0).sum() / feature_matrix.size * 100:.2f}%")
    logger.info(f"零值数量: {(feature_matrix == 0).sum()}")
    logger.info(f"零值比例: {(feature_matrix == 0).sum() / feature_matrix.size * 100:.2f}%")
    
    # 输出分位数统计
    percentiles = [0, 1, 5, 25, 50, 75, 95, 99, 100]
    logger.info("\n=== 特征值分位数统计 ===")
    for p in percentiles:
        value = np.percentile(feature_matrix, p)
        logger.info(f"{p}分位数: {value:.2f}")

def main():
    # 设置路径
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data'
    processed_dir = data_dir / 'processed'
    
    # 设置输入输出文件路径
    input_file = processed_dir / 'aligned_spectra.csv'
    output_file = processed_dir / 'feature_matrix.csv'
    
    # 检查输入文件是否存在
    if not input_file.exists():
        logger.error(f"输入文件 {input_file} 不存在")
        return
    
    # 构建特征矩阵（默认不进行PCA降维）
    build_feature_matrix(input_file, output_file, reduce_dim=False)
    
    # 如果需要降维，可以设置reduce_dim=True，并指定n_components
    # 例如：build_feature_matrix(input_file, output_file, reduce_dim=True, n_components=100)

if __name__ == "__main__":
    main()