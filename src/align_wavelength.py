#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
from pathlib import Path
import logging
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def align_wavelengths(input_file, output_file, min_wavelength=3700, max_wavelength=9000, step=1):
    """
    对光谱数据进行波长对齐
    
    参数:
    input_file: 输入CSV文件路径
    output_file: 输出CSV文件路径
    min_wavelength: 最小波长（Å）
    max_wavelength: 最大波长（Å）
    step: 波长步长（Å）
    """
    # 读取数据
    logger.info(f"正在读取数据文件: {input_file}")
    df = pd.read_csv(input_file)
    
    # 获取所有列名
    flux_cols = [col for col in df.columns if col.startswith('flux_')]
    ivar_cols = [col for col in df.columns if col.startswith('ivar_')]
    wavelength_cols = [col for col in df.columns if col.startswith('wavelength_')]
    
    # 创建新的波长网格
    new_wavelengths = np.arange(min_wavelength, max_wavelength + step, step)
    logger.info(f"创建新的波长网格: {len(new_wavelengths)} 个点")
    
    # 创建新的DataFrame来存储处理后的数据
    aligned_df = pd.DataFrame()
    aligned_df['obsid'] = df['obsid']
    aligned_df['type'] = df['type']
    
    # 统计信息
    total_spectra = len(df)
    valid_spectra = 0
    removed_spectra = 0
    
    logger.info(f"开始处理 {total_spectra} 条光谱数据...")
    
    # 对每个光谱进行处理
    for idx, row in df.iterrows():
        try:
            # 获取原始波长和通量数据
            wavelengths = row[wavelength_cols].values.astype(float)
            fluxes = row[flux_cols].values.astype(float)
            ivars = row[ivar_cols].values.astype(float)
            
            # 确保数据长度一致
            min_length = min(len(wavelengths), len(fluxes), len(ivars))
            wavelengths = wavelengths[:min_length]
            fluxes = fluxes[:min_length]
            ivars = ivars[:min_length]
            
            # 移除NaN值
            mask = ~np.isnan(wavelengths) & ~np.isnan(fluxes) & ~np.isnan(ivars)
            wavelengths = wavelengths[mask]
            fluxes = fluxes[mask]
            ivars = ivars[mask]
            
            # 检查是否有足够的数据点进行插值
            if len(wavelengths) < 2:
                removed_spectra += 1
                continue
            
            # 创建插值函数
            flux_interp = interp1d(wavelengths, fluxes, kind='linear', bounds_error=False, fill_value=np.nan)
            ivar_interp = interp1d(wavelengths, ivars, kind='linear', bounds_error=False, fill_value=np.nan)
            
            # 进行插值
            new_fluxes = flux_interp(new_wavelengths)
            new_ivars = ivar_interp(new_wavelengths)
            
            # 检查插值结果
            if np.any(np.isnan(new_fluxes)) or np.any(np.isnan(new_ivars)):
                removed_spectra += 1
                continue
            
            # 添加到新的DataFrame
            for i, wl in enumerate(new_wavelengths):
                aligned_df.loc[idx, f'wavelength_{i}'] = wl
                aligned_df.loc[idx, f'flux_{i}'] = new_fluxes[i]
                aligned_df.loc[idx, f'ivar_{i}'] = new_ivars[i]
            
            valid_spectra += 1
            
        except Exception as e:
            logger.warning(f"处理光谱 {row['obsid']} 时出错: {str(e)}")
            removed_spectra += 1
            continue
    
    # 保存处理后的数据
    logger.info("正在保存处理后的数据...")
    aligned_df.to_csv(output_file, index=False)
    
    # 输出统计信息
    logger.info("\n=== 处理完成 ===")
    logger.info(f"总光谱数量: {total_spectra}")
    logger.info(f"有效光谱数量: {valid_spectra}")
    logger.info(f"移除的光谱数量: {removed_spectra}")
    logger.info(f"保留率: {(valid_spectra/total_spectra*100):.2f}%")
    
    # 按恒星类型统计
    logger.info("\n=== 按恒星类型统计 ===")
    for star_type in ['A', 'F', 'G']:
        type_mask = aligned_df['type'] == star_type
        logger.info(f"\n{star_type}型星:")
        logger.info(f"光谱数量: {type_mask.sum()}")
        logger.info(f"保留率: {(type_mask.sum()/len(df[df['type']==star_type])*100):.2f}%")
    
    logger.info(f"\n处理后的数据已保存到: {output_file}")

def main():
    # 设置路径
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data'
    processed_dir = data_dir / 'processed'
    
    # 设置输入输出文件路径
    input_file = processed_dir / 'normalized_filtered_spectra.csv'
    output_file = processed_dir / 'aligned_spectra.csv'
    
    # 检查输入文件是否存在
    if not input_file.exists():
        logger.error(f"输入文件 {input_file} 不存在")
        return
    
    # 处理数据
    align_wavelengths(input_file, output_file)

if __name__ == "__main__":
    main() 