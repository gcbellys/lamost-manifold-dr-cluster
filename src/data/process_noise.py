#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
from pathlib import Path
import logging
import pandas as pd
import numpy as np
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_spectra_noise(input_file, output_file, threshold_ratio=0.2):
    """
    处理光谱数据中的噪声
    
    参数:
    input_file: 输入CSV文件路径
    output_file: 输出CSV文件路径
    threshold_ratio: 阈值比例，默认为0.2（20%）
    """
    # 读取数据
    logger.info(f"正在读取数据文件: {input_file}")
    df = pd.read_csv(input_file)
    
    # 获取所有列名
    flux_cols = [col for col in df.columns if col.startswith('flux_')]
    ivar_cols = [col for col in df.columns if col.startswith('ivar_')]
    wavelength_cols = [col for col in df.columns if col.startswith('wavelength_')]
    
    # 创建新的DataFrame来存储处理后的数据
    processed_df = df.copy()
    
    # 统计信息
    total_spectra = len(df)
    total_points = len(flux_cols)
    removed_points = 0
    
    # 对每个光谱进行处理
    for idx, row in df.iterrows():
        try:
            # 获取该光谱的逆方差数据并转换为float类型
            ivar_data = row[ivar_cols].values.astype(float)
            
            # 计算逆方差的中位数（忽略NaN值）
            ivar_median = np.nanmedian(ivar_data)
            
            # 设置阈值
            threshold = ivar_median * threshold_ratio
            
            # 创建掩码，保留高于阈值的数据点
            mask = np.isfinite(ivar_data) & (ivar_data >= threshold)
            
            # 统计被移除的点数
            removed_points += np.sum(~mask)
            
            # 将低信噪比的数据点设为NaN
            for i, col in enumerate(flux_cols):
                if i < len(mask) and not mask[i]:
                    processed_df.loc[idx, col] = np.nan
                    
            for i, col in enumerate(ivar_cols):
                if i < len(mask) and not mask[i]:
                    processed_df.loc[idx, col] = np.nan
                    
            for i, col in enumerate(wavelength_cols):
                if i < len(mask) and not mask[i]:
                    processed_df.loc[idx, col] = np.nan
            
            # 每处理100个文件输出一次进度
            if (idx + 1) % 100 == 0:
                logger.info(f"已处理 {idx + 1} 个光谱")
                
        except Exception as e:
            logger.error(f"处理第 {idx + 1} 个光谱时出错: {str(e)}")
            continue
    
    # 保存处理后的数据
    processed_df.to_csv(output_file, index=False)
    
    # 计算统计信息
    total_data_points = total_spectra * total_points
    removed_percentage = (removed_points / total_data_points) * 100
    remaining_percentage = 100 - removed_percentage
    
    # 输出统计信息
    logger.info("\n=== 处理完成 ===")
    logger.info(f"总光谱数量: {total_spectra}")
    logger.info(f"每个光谱的数据点数量: {total_points}")
    logger.info(f"总数据点数量: {total_data_points}")
    logger.info(f"移除的数据点数量: {removed_points}")
    logger.info(f"移除的数据点百分比: {removed_percentage:.2f}%")
    logger.info(f"保留的数据点百分比: {remaining_percentage:.2f}%")
    logger.info(f"\n处理后的数据已保存到: {output_file}")
    
    # 输出每个光谱的保留率
    retention_rates = processed_df[flux_cols].notna().sum(axis=1) / total_points * 100
    logger.info("\n=== 光谱保留率统计 ===")
    logger.info(f"平均保留率: {retention_rates.mean():.2f}%")
    logger.info(f"最小保留率: {retention_rates.min():.2f}%")
    logger.info(f"最大保留率: {retention_rates.max():.2f}%")
    
    # 按恒星类型统计
    logger.info("\n=== 按恒星类型统计 ===")
    type_stats = {}
    for star_type in ['A', 'F', 'G']:
        type_mask = processed_df['type'] == star_type
        type_retention = retention_rates[type_mask]
        type_stats[star_type] = {
            'count': type_mask.sum(),
            'mean_retention': type_retention.mean(),
            'min_retention': type_retention.min(),
            'max_retention': type_retention.max()
        }
        logger.info(f"\n{star_type}型星:")
        logger.info(f"光谱数量: {type_stats[star_type]['count']}")
        logger.info(f"平均保留率: {type_stats[star_type]['mean_retention']:.2f}%")
        logger.info(f"最小保留率: {type_stats[star_type]['min_retention']:.2f}%")
        logger.info(f"最大保留率: {type_stats[star_type]['max_retention']:.2f}%")
    
    # 生成滤波信息文本文件
    info_file = output_file.parent / 'noise_filter_info.txt'
    with open(info_file, 'w', encoding='utf-8') as f:
        f.write("=== 噪声滤波处理信息 ===\n")
        f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"输入文件: {input_file}\n")
        f.write(f"输出文件: {output_file}\n")
        f.write(f"阈值比例: {threshold_ratio}\n\n")
        
        f.write("=== 总体统计 ===\n")
        f.write(f"总光谱数量: {total_spectra}\n")
        f.write(f"每个光谱的数据点数量: {total_points}\n")
        f.write(f"总数据点数量: {total_data_points}\n")
        f.write(f"移除的数据点数量: {removed_points}\n")
        f.write(f"移除的数据点百分比: {removed_percentage:.2f}%\n")
        f.write(f"保留的数据点百分比: {remaining_percentage:.2f}%\n\n")
        
        f.write("=== 光谱保留率统计 ===\n")
        f.write(f"平均保留率: {retention_rates.mean():.2f}%\n")
        f.write(f"最小保留率: {retention_rates.min():.2f}%\n")
        f.write(f"最大保留率: {retention_rates.max():.2f}%\n\n")
        
        f.write("=== 按恒星类型统计 ===\n")
        for star_type in ['A', 'F', 'G']:
            stats = type_stats[star_type]
            f.write(f"\n{star_type}型星:\n")
            f.write(f"光谱数量: {stats['count']}\n")
            f.write(f"平均保留率: {stats['mean_retention']:.2f}%\n")
            f.write(f"最小保留率: {stats['min_retention']:.2f}%\n")
            f.write(f"最大保留率: {stats['max_retention']:.2f}%\n")
    
    logger.info(f"\n滤波信息已保存到: {info_file}")

def main():
    # 设置路径
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data'
    processed_dir = data_dir / 'processed'
    
    # 设置输入输出文件路径
    input_file = processed_dir / 'spectra_data.csv'
    output_file = processed_dir / 'noise_filtered_spectra.csv'
    
    # 检查输入文件是否存在
    if not input_file.exists():
        logger.error(f"输入文件 {input_file} 不存在")
        return
    
    # 处理数据
    process_spectra_noise(input_file, output_file)

if __name__ == "__main__":
    main() 