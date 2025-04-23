#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
from pathlib import Path
import logging
from pylamost import lamost
import pandas as pd

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def read_obsid_file(file_path):
    """读取obsid文件"""
    return pd.read_csv(file_path, header=None)[0].tolist()

# def estimate_storage_size(num_spectra, avg_size_mb=2.5):
#     """估算所需存储空间（MB）"""
#     total_size_mb = num_spectra * avg_size_mb
#     return total_size_mb

def download_spectra(l, obsid_list, save_dir, type_name):
    """下载指定类型的光谱数据"""
    success_count = 0
    failed_count = 0
    failed_obsids = []
    
    # 创建类型特定的保存目录
    type_dir = save_dir / f'type_{type_name}'
    type_dir.mkdir(parents=True, exist_ok=True)
    
    total = len(obsid_list)
    logger.info(f"\n开始下载{type_name}型星光谱数据，共{total}个...")
    
    for i, obsid in enumerate(obsid_list, 1):
        try:
            logger.info(f"正在下载 {type_name}型星 {i}/{total}: {obsid}")
            l.downloadFits(obsid=str(obsid), savedir=str(type_dir))
            success_count += 1
        except Exception as e:
            logger.error(f"下载失败 {obsid}: {str(e)}")
            failed_count += 1
            failed_obsids.append(obsid)
            
        # 每下载10个文件输出一次进度
        if i % 10 == 0:
            logger.info(f"当前进度: {i}/{total} ({(i/total*100):.2f}%)")
    
    return {
        'success': success_count,
        'failed': failed_count,
        'failed_list': failed_obsids
    }

def main():
    # 设置路径
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data'
    processed_dir = data_dir / 'processed'
    sampled_dir = processed_dir / 'sampled'
    spectra_dir = data_dir / 'spectra'
    
    # 创建lamost实例，使用固定token，并指定数据集为DR10
    l = lamost(token='Fff8ebad612', dataset=10)
    
    # 统计总数
    total_obsids_to_download = 0
    type_obsids = {}
    
    for star_type in ['A', 'F', 'G']:
        obsid_file = sampled_dir / f'type_{star_type}_obsid.txt'
        # 读取全部 obsid
        all_obsids_for_type = read_obsid_file(obsid_file)
        type_obsids[star_type] = all_obsids_for_type
        logger.info(f"{star_type}型星: 将下载 {len(all_obsids_for_type)} 个文件")
        total_obsids_to_download += len(all_obsids_for_type)
    
    logger.info(f"\n本次运行总共需要下载 {total_obsids_to_download} 个光谱文件")
    
    # 创建保存目录
    spectra_dir.mkdir(parents=True, exist_ok=True)
    
    # 下载每种类型的光谱
    results = {}
    for star_type in ['A', 'F', 'G']:
        obsids = type_obsids[star_type] # 使用完整的列表
        if not obsids:
            logger.info(f"{star_type}型星没有需要下载的obsid，跳过。")
            continue
        results[star_type] = download_spectra(l, obsids, spectra_dir, star_type)
    
    # 输出统计结果
    logger.info("\n下载统计结果:")
    for star_type, result in results.items():
        logger.info(f"\n{star_type}型星:")
        logger.info(f"成功: {result['success']} 个")
        logger.info(f"失败: {result['failed']} 个")
        if result['failed'] > 0:
            # 恢复原始失败文件名
            failed_file = spectra_dir / f'type_{star_type}_failed_obsids.txt' 
            pd.Series(result['failed_list']).to_csv(failed_file, index=False, header=False)
            logger.info(f"失败的obsid已保存到: {failed_file}")

if __name__ == "__main__":
    main() 