#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
from pathlib import Path
import logging
import pandas as pd
import numpy as np

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_data(df):
    """
    验证数据格式和内容
    
    参数:
    df: 输入的DataFrame
    
    返回:
    bool: 数据是否有效
    """
    # 检查必要的列是否存在
    required_cols = ['obsid', 'type']
    if not all(col in df.columns for col in required_cols):
        logger.error("缺少必要的列：obsid, type")
        return False
    
    # 检查波长列
    wavelength_cols = [col for col in df.columns if col.startswith('wavelength_')]
    if not wavelength_cols:
        logger.error("未找到波长列")
        return False
    
    # 检查通量列
    flux_cols = [col for col in df.columns if col.startswith('flux_')]
    if not flux_cols:
        logger.error("未找到通量列")
        return False
    
    # 检查逆方差列
    ivar_cols = [col for col in df.columns if col.startswith('ivar_')]
    if not ivar_cols:
        logger.error("未找到逆方差列")
        return False
    
    # 检查数据是否为空
    if df.empty:
        logger.error("数据为空")
        return False
    
    # 检查每行数据的有效性
    valid_rows = 0
    for idx, row in df.iterrows():
        try:
            # 转换数据类型为float
            wavelengths = pd.to_numeric(row[wavelength_cols], errors='coerce').values
            fluxes = pd.to_numeric(row[flux_cols], errors='coerce').values
            ivars = pd.to_numeric(row[ivar_cols], errors='coerce').values
            
            # 检查是否有非空值
            if not np.any(~np.isnan(wavelengths)) or not np.any(~np.isnan(fluxes)) or not np.any(~np.isnan(ivars)):
                logger.warning(f"行 {idx} 的所有数据都为空")
                continue
            
            valid_rows += 1
        except Exception as e:
            logger.warning(f"处理行 {idx} 时出错: {str(e)}")
            continue
    
    if valid_rows == 0:
        logger.error("没有有效的数据行")
        return False
    
    logger.info(f"数据验证通过，有效行数：{valid_rows}")
    return True

def normalize_flux_data(input_file, output_file):
    """
    使用z-score标准化方法处理通量数据
    
    参数:
    input_file: 输入CSV文件路径
    output_file: 输出CSV文件路径
    """
    # 读取数据
    logger.info(f"正在读取数据文件: {input_file}")
    df = pd.read_csv(input_file)
    
    # 验证数据
    if not validate_data(df):
        logger.error("数据验证失败，程序终止")
        return
    
    # 获取所有列名
    flux_cols = [col for col in df.columns if col.startswith('flux_')]
    ivar_cols = [col for col in df.columns if col.startswith('ivar_')]
    wavelength_cols = [col for col in df.columns if col.startswith('wavelength_')]
    
    # 创建新的DataFrame来存储处理后的数据
    normalized_df = df.copy()
    
    # 统计信息
    total_spectra = len(df)
    total_points = len(flux_cols)
    
    logger.info(f"总光谱数量: {total_spectra}")
    logger.info(f"每个光谱的数据点数量: {total_points}")
    
    # 对每个光谱进行处理
    for idx, row in df.iterrows():
        try:
            # 获取该光谱的通量数据并转换为float类型
            flux_data = pd.to_numeric(row[flux_cols], errors='coerce').values
            
            # 计算均值和标准差（忽略NaN值）
            flux_mean = np.nanmean(flux_data)
            flux_std = np.nanstd(flux_data)
            
            # 标准化通量数据
            if flux_std != 0:  # 避免除以零
                normalized_flux = (flux_data - flux_mean) / flux_std
            else:
                normalized_flux = flux_data - flux_mean
            
            # 更新数据
            normalized_df.loc[idx, flux_cols] = normalized_flux
            
            # 每处理100个光谱输出一次进度
            if (idx + 1) % 100 == 0:
                logger.info(f"已处理 {idx + 1} 个光谱")
                
        except Exception as e:
            logger.warning(f"处理光谱 {row['obsid']} 时出错: {str(e)}")
            continue
    
    # 保存处理后的数据
    logger.info("正在保存处理后的数据...")
    normalized_df.to_csv(output_file, index=False)
    
    # 输出统计信息
    logger.info("\n=== 处理完成 ===")
    logger.info(f"总光谱数量: {total_spectra}")
    logger.info(f"每个光谱的数据点数量: {total_points}")
    
    # 计算并输出标准化后的统计信息
    all_normalized_flux = normalized_df[flux_cols].values.flatten()
    all_normalized_flux = all_normalized_flux[~np.isnan(all_normalized_flux)]
    
    logger.info("\n=== 标准化后的统计信息 ===")
    logger.info(f"均值: {np.mean(all_normalized_flux):.6f}")
    logger.info(f"标准差: {np.std(all_normalized_flux):.6f}")
    logger.info(f"最小值: {np.min(all_normalized_flux):.6f}")
    logger.info(f"最大值: {np.max(all_normalized_flux):.6f}")
    
    # 按恒星类型统计
    logger.info("\n=== 按恒星类型统计 ===")
    for star_type in ['A', 'F', 'G']:
        type_mask = normalized_df['type'] == star_type
        type_flux = normalized_df.loc[type_mask, flux_cols].values.flatten()
        type_flux = type_flux[~np.isnan(type_flux)]
        
        logger.info(f"\n{star_type}型星:")
        logger.info(f"光谱数量: {type_mask.sum()}")
        logger.info(f"均值: {np.mean(type_flux):.6f}")
        logger.info(f"标准差: {np.std(type_flux):.6f}")
        logger.info(f"最小值: {np.min(type_flux):.6f}")
        logger.info(f"最大值: {np.max(type_flux):.6f}")
    
    logger.info(f"\n处理后的数据已保存到: {output_file}")

def main():
    # 设置路径
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data'
    processed_dir = data_dir / 'processed'
    
    # 设置输入输出文件路径
    input_file = processed_dir / 'noise_filtered_spectra.csv'
    output_file = processed_dir / 'normalized_filtered_spectra.csv'
    
    # 检查输入文件是否存在
    if not input_file.exists():
        logger.error(f"输入文件 {input_file} 不存在")
        return
    
    # 处理数据
    normalize_flux_data(input_file, output_file)

if __name__ == "__main__":
    main() 