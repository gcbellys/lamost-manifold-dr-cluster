#!/usr/bin/python
# -*- coding: UTF-8 -*-

import argparse
import pandas as pd
import numpy as np

def check_csv_format(file_path):
    """检查CSV文件的格式"""
    # 读取CSV文件
    df = pd.read_csv(file_path)
    
    # 检查基本结构
    print("\n=== 基本结构检查 ===")
    print(f"总行数（光谱数量）: {len(df)}")
    print(f"总列数: {len(df.columns)}")
    print(f"每个光谱的波长点数: {len([col for col in df.columns if col.startswith('wavelength_')])}")
    
    # 按类型统计行数
    type_counts = df['type'].value_counts()
    print("\n各类型光谱数量:")
    for star_type, count in type_counts.items():
        print(f"- {star_type}型星: {count} 条光谱")
    
    # 检查数据类型
    print("\n=== 数据类型检查 ===")
    print(df.dtypes)
    
    # 检查数据范围
    print("\n=== 数据范围检查 ===")
    print("波长范围:")
    wavelength_cols = [col for col in df.columns if col.startswith('wavelength_')]
    if wavelength_cols:
        min_wl = df[wavelength_cols].min().min()
        max_wl = df[wavelength_cols].max().max()
        print(f"最小值: {min_wl:.2f}")
        print(f"最大值: {max_wl:.2f}")
    
    print("\n通量范围:")
    flux_cols = [col for col in df.columns if col.startswith('flux_')]
    if flux_cols:
        min_flux = df[flux_cols].min().min()
        max_flux = df[flux_cols].max().max()
        print(f"最小值: {min_flux:.2f}")
        print(f"最大值: {max_flux:.2f}")
    
    print("\n逆方差范围:")
    ivar_cols = [col for col in df.columns if col.startswith('ivar_')]
    if ivar_cols:
        min_ivar = df[ivar_cols].min().min()
        max_ivar = df[ivar_cols].max().max()
        print(f"最小值: {min_ivar:.2f}")
        print(f"最大值: {max_ivar:.2f}")
    
    # 检查缺失值
    print("\n=== 缺失值检查 ===")
    missing_values = df.isnull().sum()
    missing_values = missing_values[missing_values > 0]
    if not missing_values.empty:
        print(f"存在缺失值的列数: {len(missing_values)}")
        print(f"缺失值最多的前5列:")
        print(missing_values.nlargest(5))

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='检查CSV文件格式')
    parser.add_argument('input_file', help='CSV文件路径')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 检查文件格式
    check_csv_format(args.input_file)

if __name__ == "__main__":
    main()