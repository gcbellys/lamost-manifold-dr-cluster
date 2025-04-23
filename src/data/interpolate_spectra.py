#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import logging
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def interpolate_spectra(input_file, output_file):
    """
    对光谱数据进行插值，将不同波长的光谱对齐到相同的波长网格
    
    参数:
    input_file: 输入CSV文件路径
    output_file: 输出CSV文件路径
    """
    # 读取数据
    logger.info(f"正在读取文件: {input_file}")
    try:
        data = pd.read_csv(input_file)
    except Exception as e:
        logger.error(f"读取文件失败: {str(e)}")
        return
    
    # 定义列名
    obsid_col = 'obsid'
    type_col = 'type'
    wavelength_cols = [col for col in data.columns if col.startswith('wavelength_')]
    flux_cols = [col for col in data.columns if col.startswith('flux_')]
    ivar_cols = [col for col in data.columns if col.startswith('ivar_')]
    
    # 检查必要列
    if not (wavelength_cols and flux_cols and ivar_cols):
        logger.error("缺少必要的波长、通量或逆方差列")
        return
    
    # 提取数据
    obsids = data[obsid_col]
    types = data[type_col]
    wavelengths = data[wavelength_cols].values
    fluxes = data[flux_cols].values
    ivars = data[ivar_cols].values
    
    # 初始化插值后的数据列表
    interpolated_flux_list = []
    interpolated_ivar_list = []
    valid_obsids = []
    valid_types = []
    
    # 统计信息
    total_spectra = len(data)
    skipped_spectra = 0
    invalid_range = 0
    invalid_interp = 0
    negative_ivar = 0
    
    # 处理每个光谱
    for idx in range(len(data)):
        try:
            obsid = obsids[idx]
            spectrum_type = types[idx]
            wl = wavelengths[idx]
            flux = fluxes[idx]
            ivar = ivars[idx]
            
            # 剔除包含 NaN 的点
            mask = ~np.isnan(wl) & ~np.isnan(flux) & ~np.isnan(ivar)
            wl = wl[mask]
            flux = flux[mask]
            ivar = ivar[mask]
            
            # 输出处理信息
            logger.info(f"正在处理 obsid: {obsid}, 类型: {spectrum_type}, 有效光谱量: {len(wl)}")
            
            # 检查有效数据点数量
            if len(wl) < 3000:  # 设置最小有效数据点数量
                logger.warning(f"obsid {obsid}: 有效数据点数量过少 ({len(wl)}), 跳过")
                skipped_spectra += 1
                invalid_range += 1
                continue
            
            # 检查波长范围
            if len(wl) == 0:
                logger.warning(f"obsid {obsid}: 空数据, 跳过")
                skipped_spectra += 1
                invalid_range += 1
                continue
                
            # 获取实际波长范围
            min_wl = np.ceil(wl.min())
            max_wl = np.floor(wl.max())
            
            # 如果波长范围太小，跳过
            if max_wl - min_wl < 1000:  # 至少需要1000Å的波长范围
                logger.warning(f"obsid {obsid}: 波长范围太小 ({min_wl:.1f}–{max_wl:.1f}Å), 跳过")
                skipped_spectra += 1
                invalid_range += 1
                continue
            
            # 创建目标波长网格，使用实际波长范围
            target_wavelength = np.arange(min_wl, max_wl + 1, 1)
            
            # 线性插值，严格限制在波长范围内
            flux_interp = interp1d(wl, flux, kind='linear', bounds_error=True, fill_value=None)
            ivar_interp = interp1d(wl, ivar, kind='linear', bounds_error=True, fill_value=None)
            
            try:
                interpolated_flux = flux_interp(target_wavelength)
                interpolated_ivar = ivar_interp(target_wavelength)
            except ValueError as e:
                logger.warning(f"obsid {obsid}: 插值失败 - {str(e)}, 跳过")
                skipped_spectra += 1
                invalid_interp += 1
                continue
            
            # 检查插值结果中的 NaN 或 inf
            if np.any(np.isnan(interpolated_flux)) or np.any(np.isnan(interpolated_ivar)) or \
               np.any(np.isinf(interpolated_flux)) or np.any(np.isinf(interpolated_ivar)):
                logger.warning(f"obsid {obsid}: 插值后包含 NaN 或 inf, 跳过")
                skipped_spectra += 1
                invalid_interp += 1
                continue
            
            # 检查负逆方差值
            if np.any(interpolated_ivar < 0):
                logger.warning(f"obsid {obsid}: 插值后逆方差值包含负值 ({np.sum(interpolated_ivar < 0)} 个), 跳过")
                skipped_spectra += 1
                negative_ivar += 1
                continue
            
            # 检查负通量值并记录
            negative_flux_count = np.sum(interpolated_flux < 0)
            if negative_flux_count > 0:
                logger.warning(f"obsid {obsid}: 插值后通量包含负值 ({negative_flux_count} 个, 占比 {negative_flux_count/len(interpolated_flux)*100:.2f}%)")
            
            interpolated_flux_list.append(interpolated_flux)
            interpolated_ivar_list.append(interpolated_ivar)
            valid_obsids.append(obsid)
            valid_types.append(spectrum_type)
            
        except Exception as e:
            logger.error(f"处理第 {idx + 1} 个光谱时出错: {str(e)}")
            skipped_spectra += 1
            continue
    
    # 构建对齐后的数据
    if interpolated_flux_list:
        # 确保所有数组具有相同的长度
        max_length = max(len(arr) for arr in interpolated_flux_list)
        interpolated_flux_array = np.zeros((len(interpolated_flux_list), max_length))
        interpolated_ivar_array = np.zeros((len(interpolated_ivar_list), max_length))
        
        for i, (flux, ivar) in enumerate(zip(interpolated_flux_list, interpolated_ivar_list)):
            interpolated_flux_array[i, :len(flux)] = flux
            interpolated_ivar_array[i, :len(ivar)] = ivar
            
        logger.info(f"成功插值 {len(valid_obsids)} 个光谱")
    else:
        logger.error("没有光谱成功插值，请检查输入数据")
        return
    
    # 保存对齐后的数据
    target_wavelength = np.arange(min_wl, min_wl + max_length, 1)
    columns = ['obsid', 'type'] + [f'flux_{wl:.1f}' for wl in target_wavelength] + [f'ivar_{wl:.1f}' for wl in target_wavelength]
    df_output = pd.DataFrame({
        'obsid': valid_obsids,
        'type': valid_types
    })
    df_output = pd.concat([df_output, pd.DataFrame(interpolated_flux_array, columns=[f'flux_{wl:.1f}' for wl in target_wavelength])], axis=1)
    df_output = pd.concat([df_output, pd.DataFrame(interpolated_ivar_array, columns=[f'ivar_{wl:.1f}' for wl in target_wavelength])], axis=1)
    
    # 最终数据质量检查
    flux_columns = [f'flux_{wl:.1f}' for wl in target_wavelength]
    ivar_columns = [f'ivar_{wl:.1f}' for wl in target_wavelength]
    negative_ivar_count = (df_output[ivar_columns] < 0).sum().sum()
    if negative_ivar_count > 0:
        logger.error(f"最终数据中仍包含 {negative_ivar_count} 个负逆方差值，终止保存")
        return
    
    negative_flux_count = (df_output[flux_columns] < 0).sum().sum()
    if negative_flux_count > 0:
        logger.warning(f"最终数据中包含 {negative_flux_count} 个负通量值，占比 {negative_flux_count/(len(df_output)*len(flux_columns))*100:.2f}%")
    
    df_output.to_csv(output_file, index=False)
    
    # 输出统计信息
    logger.info("\n=== 处理完成 ===")
    logger.info(f"总光谱数量: {total_spectra}")
    logger.info(f"成功插值数量: {len(valid_obsids)}")
    logger.info(f"跳过数量: {skipped_spectra}")
    logger.info(f"波长范围不完整: {invalid_range}")
    logger.info(f"插值结果无效: {invalid_interp}")
    logger.info(f"负逆方差值: {negative_ivar}")
    
    # 按恒星类型统计
    logger.info("\n=== 按恒星类型统计 ===")
    for star_type in ['A', 'F', 'G']:
        type_count = len(df_output[df_output['type'] == star_type])
        logger.info(f"{star_type}型星: {type_count} 个")
    
    logger.info(f"\n对齐后的数据已保存至 {output_file}")

def main():
    # 设置路径
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data'
    processed_dir = data_dir / 'processed'
    log_dir = Path('/home/lamost-manifold-dr-cluster/log')
    
    # 确保日志目录存在
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置输入输出文件路径
    input_file = processed_dir / 'normalized_filtered_spectra.csv'
    output_file = processed_dir / 'aligned_spectra.csv'
    
    # 设置日志文件路径
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'interpolate_spectra_{timestamp}.log'
    
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
    
    # 处理数据
    interpolate_spectra(input_file, output_file)
    
    # 移除文件日志处理器
    logger.removeHandler(file_handler)
    logger.info(f"日志已保存至 {log_file}")

if __name__ == "__main__":
    main()