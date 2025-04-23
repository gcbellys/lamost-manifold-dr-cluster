#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
from pathlib import Path
import pandas as pd
import numpy as np
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_feature_matrix(input_file):
    """
    分析特征矩阵中负数特征值的数量
    
    参数:
    input_file: 输入CSV文件路径（特征矩阵）
    """
    # 加载数据
    logger.info(f"正在加载文件: {input_file}")
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        logger.error(f"加载文件失败: {str(e)}")
        return
    
    # 提取特征列（排除obsid和type列）
    feature_columns = [col for col in df.columns if col not in ['obsid', 'type']]
    feature_df = df[feature_columns]
    
    # 统计负数特征值
    negative_counts = (feature_df < 0).sum()
    total_negative = negative_counts.sum()
    total_values = feature_df.size
    
    # 输出统计信息
    logger.info("\n=== 负数特征值统计 ===")
    logger.info(f"总特征值数量: {total_values}")
    logger.info(f"负数特征值数量: {total_negative}")
    logger.info(f"负数特征值比例: {(total_negative/total_values)*100:.2f}%")
    
    # 按列统计负数特征值
    logger.info("\n=== 按列统计负数特征值 ===")
    for col in feature_columns:
        neg_count = negative_counts[col]
        neg_ratio = (neg_count / len(df)) * 100
        logger.info(f"{col}: {neg_count}个负数 ({neg_ratio:.2f}%)")
    
    # 按光谱类型统计负数特征值
    logger.info("\n=== 按光谱类型统计负数特征值 ===")
    for star_type in df['type'].unique():
        type_df = df[df['type'] == star_type]
        type_feature_df = type_df[feature_columns]
        type_negative = (type_feature_df < 0).sum().sum()
        type_total = type_feature_df.size
        type_ratio = (type_negative / type_total) * 100
        logger.info(f"{star_type}型星: {type_negative}个负数 ({type_ratio:.2f}%)")

def main():
    # 设置路径
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data'
    processed_dir = data_dir / 'processed'
    
    # 设置输入文件路径
    input_file = processed_dir / 'feature_matrix.csv'
    
    # 检查输入文件是否存在
    if not input_file.exists():
        logger.error(f"输入文件 {input_file} 不存在")
        return
    
    # 分析特征矩阵
    analyze_feature_matrix(input_file)

if __name__ == "__main__":
    main() 