#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
from pathlib import Path
import logging
import astropy.io.fits as fits
import pandas as pd
import numpy as np

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def filter_low_snr_data(flux, ivar, wavelength, threshold_ratio=0.2):
    """
    使用逆方差进行滤波，剔除低信噪比的数据点
    
    参数:
    flux: 通量数据
    ivar: 逆方差数据
    wavelength: 波长数据
    threshold_ratio: 阈值比例，默认为0.2（20%）
    
    返回:
    过滤后的通量、逆方差和波长数据
    """
    # 计算逆方差的中位数
    ivar_median = np.median(ivar)
    
    # 设置阈值
    threshold = ivar_median * threshold_ratio
    
    # 创建掩码，保留高于阈值的数据点
    mask = ivar >= threshold
    
    # 应用掩码
    filtered_flux = flux[mask]
    filtered_ivar = ivar[mask]
    filtered_wavelength = wavelength[mask]
    
    return filtered_flux, filtered_ivar, filtered_wavelength


def process_spectra_data(spectra_dir, output_file):
    """处理光谱数据并保存为CSV文件"""
    # 创建列表来存储所有数据
    all_data = []
    processed_obsids = set()  # 用于跟踪已处理的观测ID
    
    # 遍历A、F、G三个类型的文件夹
    for star_type in ['A', 'F', 'G']:
        type_dir = spectra_dir / f'type_{star_type}'
        if not type_dir.exists():
            logger.warning(f"目录 {type_dir} 不存在，跳过")
            continue
            
        logger.info(f"正在处理 {star_type} 型星光谱数据...")
        
        # 遍历该类型下的所有fits.gz文件
        for fits_file in type_dir.glob('*.fits.gz'):  # 只处理.fits.gz文件
            try:
                # 读取FITS文件
                with fits.open(fits_file) as hdul:
                    # 获取观测ID（在第一个HDU的header中）
                    obsid = hdul[0].header['OBSID']
                    
                    # 检查是否已经处理过这个观测ID
                    if obsid in processed_obsids:
                        logger.warning(f"跳过重复的观测ID: {obsid}")
                        continue
                    
                    # 获取光谱数据（在第二个HDU中）
                    data = hdul[1].data
                    flux = data['FLUX'][0]      # 通量
                    ivar = data['IVAR'][0]      # 逆方差
                    wavelength = data['WAVELENGTH'][0]  # 波长
                    
                    # 创建该文件的数据字典
                    file_data = {
                        'obsid': obsid,
                        'type': star_type,
                        'wavelength': wavelength,
                        'flux': flux,
                        'ivar': ivar
                    }
                    
                    # 添加到总数据中
                    all_data.append(file_data)
                    processed_obsids.add(obsid)  # 记录已处理的观测ID
                    
                    # 每处理100个文件输出一次进度
                    if len(all_data) % 100 == 0:
                        logger.info(f"已处理 {len(all_data)} 个文件")
                    
            except Exception as e:
                logger.error(f"处理文件 {fits_file} 时出错: {str(e)}")
                continue
    
    if not all_data:
        logger.warning("没有找到任何有效的光谱数据")
        return
    
    # 创建新的DataFrame
    new_data = []
    for data in all_data:
        row = {
            'obsid': data['obsid'],
            'type': data['type']
        }
        
        # 添加每个光谱自身的波长、通量和逆方差
        for i in range(len(data['wavelength'])):
            row[f'wavelength_{i}'] = data['wavelength'][i]
            row[f'flux_{i}'] = data['flux'][i]
            row[f'ivar_{i}'] = data['ivar'][i]
        
        new_data.append(row)
    
    # 创建DataFrame并保存
    new_df = pd.DataFrame(new_data)
    new_df.to_csv(output_file, index=False)
    logger.info(f"数据已保存到 {output_file}")
    
    # 输出数据形状信息
    logger.info(f"\n数据形状: {new_df.shape}")
    logger.info(f"波长点数量: {len(data['wavelength'])}")
    logger.info(f"光谱数量: {len(new_data)}")
    
    # 按恒星类型统计
    logger.info("\n=== 按恒星类型统计 ===")
    for star_type in ['A', 'F', 'G']:
        type_count = len(new_df[new_df['type'] == star_type])
        logger.info(f"{star_type}型星: {type_count} 个")

def main():
    # 设置路径
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data'
    processed_dir = data_dir / 'processed'
    spectra_dir = data_dir / 'spectra'
    
    # 处理光谱数据并保存为CSV
    output_file = processed_dir / 'spectra_data.csv'
    process_spectra_data(spectra_dir, output_file)

if __name__ == "__main__":
    main()