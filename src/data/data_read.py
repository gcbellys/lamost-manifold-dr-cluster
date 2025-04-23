from astropy.io import fits
import pandas as pd
import numpy as np
import os
from pathlib import Path
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_directories(base_path, dirs):
    """创建必要的目录"""
    for d in dirs:
        path = base_path / d
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {path}")

def main():
    # 设置路径
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data'
    processed_dir = data_dir / 'processed'
    
    # 创建保存目录
    create_directories(data_dir, ['processed'])
    
    # 读取FITS文件
    logger.info("Reading FITS file...")
    fits_path = data_dir / 'raw' / 'dr10_v0_LRS_stellar_q1q2q3.fits'
    
    try:
        with fits.open(fits_path) as hdul:
            data = hdul[1].data
            logger.info(f"Total spectra in catalog: {len(data)}")
            
            # 转换为DataFrame
            df = pd.DataFrame(data.tolist(), columns=data.names)
            
            # 筛选A/F/G型星且SNR > 10的目标
            filtered = df[
                (df['class'] == 'STAR') &  # 确保是恒星
                (df['subclass'].str[0].isin(['A', 'F', 'G'])) &  # 匹配首字母为A、F、G的子类型
                (df['snrg'] > 10)
            ]
            
            logger.info(f"筛选后的光谱数量: {len(filtered)}")
            logger.info("\n各光谱类型数量:")
            logger.info("\n" + str(filtered['subclass'].value_counts().head(10)))
            
            # 保存obsid列表
            obsid_file = processed_dir / 'AFG_obsid.txt'
            filtered['obsid'].to_csv(obsid_file, index=False, header=False)
            logger.info(f"筛选后的obsid列表已保存到: {obsid_file}")
            
            # 保存基本参数
            params_file = processed_dir / 'AFG_params.csv'
            columns = ['obsid', 'class', 'subclass', 'snrg', 'teff', 'logg', 'feh', 'ra', 'dec', 'z']
            filtered[columns].to_csv(params_file, index=False)
            logger.info(f"基本参数已保存到: {params_file}")
            
            # 输出一些统计信息
            logger.info("\n基本参数统计:")
            stats = filtered[['teff', 'logg', 'feh', 'snrg']].describe()
            logger.info("\n" + str(stats))
            
    except FileNotFoundError:
        logger.error(f"Error: Could not find file {fits_path}")
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 