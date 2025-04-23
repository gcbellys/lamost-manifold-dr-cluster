import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# 设置绘图样式
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_context("paper", font_scale=1.2)

# 创建保存图片的目录
base_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(base_dir)
data_dir = os.path.join(project_dir, 'data')
figures_dir = os.path.join(data_dir, 'figures')
os.makedirs(figures_dir, exist_ok=True)

try:
    # 读取数据
    data_path = os.path.join(data_dir, 'processed', 'AFG_stars.csv')
    print(f"Reading data file: {data_path}")
    df = pd.read_csv(data_path)
    print("Data file loaded successfully")

    # 创建一个函数来保存图片
    def save_fig(fig, name):
        save_path = os.path.join(figures_dir, f'{name}.png')
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure: {save_path}")
        plt.close(fig)

    # 1. 温度分布直方图
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='teff', bins=50)
    plt.title('Stellar Effective Temperature Distribution')
    plt.xlabel('Effective Temperature (K)')
    plt.ylabel('Count')
    save_fig(plt.gcf(), 'temperature_distribution')

    # 2. 不同光谱类型的温度箱线图
    plt.figure(figsize=(12, 6))
    df['spectral_type'] = df['subclass'].str[0]
    sns.boxplot(data=df, x='spectral_type', y='teff')
    plt.title('Temperature Distribution by Spectral Type')
    plt.xlabel('Spectral Type')
    plt.ylabel('Effective Temperature (K)')
    save_fig(plt.gcf(), 'temperature_by_spectral_type')

    # 3. HR图
    plt.figure(figsize=(10, 8))
    scatter = sns.scatterplot(data=df, x='teff', y='logg', alpha=0.5, hue='spectral_type', palette='deep')
    plt.title('HR Diagram')
    plt.xlabel('Effective Temperature (K)')
    plt.ylabel('Surface Gravity log(g)')
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    scatter.legend(title='Spectral Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    save_fig(plt.gcf(), 'hr_diagram')

    # 4. 金属丰度分布
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='feh', bins=50)
    plt.title('Metallicity Distribution')
    plt.xlabel('[Fe/H]')
    plt.ylabel('Count')
    save_fig(plt.gcf(), 'metallicity_distribution')

    # 5. 相关性热图
    correlation_columns = ['teff', 'logg', 'feh', 'snrg']
    plt.figure(figsize=(8, 6))
    sns.heatmap(df[correlation_columns].corr(), annot=True, cmap='coolwarm', center=0,
                xticklabels=['Teff', 'log(g)', '[Fe/H]', 'SNR'],
                yticklabels=['Teff', 'log(g)', '[Fe/H]', 'SNR'])
    plt.title('Parameter Correlation Heatmap')
    save_fig(plt.gcf(), 'correlation_heatmap')

    # 6. 双变量分布图
    g = sns.JointGrid(data=df, x='teff', y='logg', hue='spectral_type', height=8)
    g.plot_joint(sns.scatterplot, alpha=0.5)
    g.plot_marginals(sns.histplot)
    g.fig.suptitle('Temperature-Gravity Joint Distribution', y=1.02)
    save_fig(g.figure, 'teff_logg_joint')

    # 7. 信噪比与温度的关系
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='teff', y='snrg', alpha=0.5, hue='spectral_type')
    plt.title('SNR vs Temperature')
    plt.xlabel('Effective Temperature (K)')
    plt.ylabel('Signal-to-Noise Ratio')
    save_fig(plt.gcf(), 'snr_vs_temperature')

    # 8. 各参数的小提琴图
    params = ['teff', 'logg', 'feh', 'snrg']
    param_names = ['Temperature (K)', 'Surface Gravity', 'Metallicity', 'SNR']
    fig, axes = plt.subplots(1, len(params), figsize=(20, 6))
    for ax, param, name in zip(axes, params, param_names):
        sns.violinplot(data=df, y=param, x='spectral_type', ax=ax)
        ax.set_title(name)
        ax.set_xlabel('Spectral Type')
    plt.tight_layout()
    save_fig(plt.gcf(), 'parameter_distributions')

    print(f"\nAll figures have been saved to: {figures_dir}")

    # 输出统计信息
    print("\nBasic Statistics:")
    stats = df[['teff', 'logg', 'feh', 'snrg']].describe()
    stats.index = ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']
    print(stats)

    print("\nSpectral Type Distribution:")
    type_counts = df['spectral_type'].value_counts()
    for type_name, count in type_counts.items():
        print(f"Type {type_name}: {count:,} stars ({count/len(df)*100:.1f}%)")

except FileNotFoundError as e:
    print(f"Error: File not found - {e}")
except Exception as e:
    print(f"Error: {e}")
    raise 