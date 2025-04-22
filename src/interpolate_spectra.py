#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def interpolate_spectra(input_file, output_file):
    """
    对光谱数据进行插值，将不同波长的光谱对齐到相同的波长网格
    
    参数:
    input_file: 输入CSV文件路径
    output_file: 输出CSV文件路径
    """
    # 定义目标波长网格：3700–9000Å，步长 1Å，共 5301 点
    target_wavelength = np.arange(3700, 9001, 1)
    
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
    wavelength_cols = [f'wavelength_{i}' for i in range(3909)]
    flux_cols = [f'flux_{i}' for i in range(3909)]
    ivar_cols = [f'ivar_{i}' for i in range(3909)]
    
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
            
            # 检查波长范围是否覆盖 3700–9000Å
            if len(wl) == 0 or wl.min() > 3700 or wl.max() < 9000:
                logger.warning(f"obsid {obsid}: 波长范围不完整 ({wl.min():.1f}–{wl.max():.1f}Å 或空数据), 跳过")
                skipped_spectra += 1
                invalid_range += 1
                continue
            
            # 线性插值
            flux_interp = interp1d(wl, flux, kind='linear', bounds_error=False, fill_value='extrapolate')
            ivar_interp = interp1d(wl, ivar, kind='linear', bounds_error=False, fill_value='extrapolate')
            interpolated_flux = flux_interp(target_wavelength)
            interpolated_ivar = ivar_interp(target_wavelength)
            
            # 检查插值结果中的 NaN 或 inf
            if np.any(np.isnan(interpolated_flux)) or np.any(np.isnan(interpolated_ivar)) or \
               np.any(np.isinf(interpolated_flux)) or np.any(np.isinf(interpolated_ivar)):
                logger.warning(f"obsid {obsid}: 插值后包含 NaN 或 inf, 跳过")
                skipped_spectra += 1
                invalid_interp += 1
                continue
            
            interpolated_flux_list.append(interpolated_flux)
            interpolated_ivar_list.append(interpolated_ivar)
            valid_obsids.append(obsid)
            valid_types.append(spectrum_type)
            
            # 每处理100个文件输出一次进度
            if (idx + 1) % 100 == 0:
                logger.info(f"已处理 {idx + 1} 个光谱")
                
        except Exception as e:
            logger.error(f"处理第 {idx + 1} 个光谱时出错: {str(e)}")
            skipped_spectra += 1
            continue
    
    # 构建对齐后的数据
    if interpolated_flux_list:
        interpolated_flux_array = np.array(interpolated_flux_list)
        interpolated_ivar_array = np.array(interpolated_ivar_list)
        logger.info(f"成功插值 {len(valid_obsids)} 个光谱")
    else:
        logger.error("没有光谱成功插值，请检查输入数据")
        return
    
    # 保存对齐后的数据
    columns = ['obsid', 'type'] + [f'flux_{wl:.1f}' for wl in target_wavelength] + [f'ivar_{wl:.1f}' for wl in target_wavelength]
    df_output = pd.DataFrame({
        'obsid': valid_obsids,
        'type': valid_types
    })
    df_output = pd.concat([df_output, pd.DataFrame(interpolated_flux_array, columns=[f'flux_{wl:.1f}' for wl in target_wavelength])], axis=1)
    df_output = pd.concat([df_output, pd.DataFrame(interpolated_ivar_array, columns=[f'ivar_{wl:.1f}' for wl in target_wavelength])], axis=1)
    df_output.to_csv(output_file, index=False)
    
    # 输出统计信息
    logger.info("\n=== 处理完成 ===")
    logger.info(f"总光谱数量: {total_spectra}")
    logger.info(f"成功插值数量: {len(valid_obsids)}")
    logger.info(f"跳过数量: {skipped_spectra}")
    logger.info(f"波长范围不完整: {invalid_range}")
    logger.info(f"插值结果无效: {invalid_interp}")
    
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
    
    # 设置输入输出文件路径
    input_file = processed_dir / 'noise_filtered_spectra.csv'
    output_file = processed_dir / 'aligned_spectra.csv'
    
    # 检查输入文件是否存在
    if not input_file.exists():
        logger.error(f"输入文件 {input_file} 不存在")
        return
    
    # 处理数据
    interpolate_spectra(input_file, output_file)

if __name__ == "__main__":
    main()