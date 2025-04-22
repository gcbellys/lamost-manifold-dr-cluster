import pandas as pd
import numpy as np
from pathlib import Path
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def sample_by_type(df, type_col, n_samples=1000):
    """从每个类型中随机抽取n_samples条数据，并返回每种类型的数据和合并后的数据"""
    # 获取所有类型
    types = df[type_col].str[0].unique()
    sampled_data = {}
    
    for star_type in types:
        # 获取该类型的所有数据
        type_data = df[df[type_col].str[0] == star_type]
        # 计算实际可以抽取的样本数（取最小值）
        actual_samples = min(n_samples, len(type_data))
        # 随机抽样
        sampled = type_data.sample(n=actual_samples, random_state=42)
        sampled_data[star_type] = sampled
        logger.info(f"{star_type}型星：总数{len(type_data)}，抽取{actual_samples}条")
    
    # 合并所有抽样数据
    merged_data = pd.concat(list(sampled_data.values()), ignore_index=True)
    return sampled_data, merged_data

def save_type_data(data_dict, base_dir):
    """保存每种类型的数据到单独的文件"""
    for star_type, df in data_dict.items():
        # 保存参数文件
        params_file = base_dir / f'type_{star_type}_params.csv'
        df.to_csv(params_file, index=False)
        logger.info(f"{star_type}型星数据已保存到: {params_file}")
        
        # 保存obsid列表
        obsid_file = base_dir / f'type_{star_type}_obsid.txt'
        df['obsid'].to_csv(obsid_file, index=False, header=False)
        logger.info(f"{star_type}型星obsid列表已保存到: {obsid_file}")

def main():
    # 设置路径
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data'
    processed_dir = data_dir / 'processed'
    sample_dir = processed_dir / 'sampled'
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 读取原始数据
        params_file = processed_dir / 'AFG_params.csv'
        logger.info(f"Reading data from: {params_file}")
        df = pd.read_csv(params_file)
        logger.info(f"Total records: {len(df)}")
        
        # 进行抽样
        logger.info("\n开始随机抽样...")
        sampled_dict, merged_df = sample_by_type(df, 'subclass', n_samples=1000)
        
        # 保存每种类型的数据
        logger.info("\n保存分类数据...")
        save_type_data(sampled_dict, sample_dir)
        
        # 保存合并的数据
        merged_params_file = sample_dir / 'AFG_merged_params.csv'
        merged_df.to_csv(merged_params_file, index=False)
        logger.info(f"\n合并数据已保存到: {merged_params_file}")
        
        merged_obsid_file = sample_dir / 'AFG_merged_obsid.txt'
        merged_df['obsid'].to_csv(merged_obsid_file, index=False, header=False)
        logger.info(f"合并的obsid列表已保存到: {merged_obsid_file}")
        
        # 显示每种类型的统计信息
        logger.info("\n各类型数据统计:")
        for star_type, type_df in sampled_dict.items():
            logger.info(f"\n{star_type}型星统计信息:")
            stats = type_df[['teff', 'logg', 'feh', 'snrg']].describe()
            logger.info("\n" + str(stats))
        
        # 显示合并数据的统计信息
        logger.info("\n合并数据统计:")
        logger.info("\n各类型数量:")
        type_counts = merged_df['subclass'].str[0].value_counts()
        for type_name, count in type_counts.items():
            logger.info(f"{type_name}型星: {count}条")
        
        logger.info("\n基本参数统计:")
        merged_stats = merged_df[['teff', 'logg', 'feh', 'snrg']].describe()
        logger.info("\n" + str(merged_stats))
        
    except FileNotFoundError:
        logger.error(f"Error: Could not find file {params_file}")
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 