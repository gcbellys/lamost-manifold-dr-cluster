import torch
import sys
import platform

# 检测Python版本
print(f"Python 版本: {sys.version}")
print(f"操作系统: {platform.system()} {platform.version()}")

# PyTorch版本
print(f"\nPyTorch 版本: {torch.__version__}")
print(f"CUDA 是否可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"当前GPU设备: {torch.cuda.get_device_name(0)}")

try:
    import numpy as np
    print(f"\nNumPy 版本: {np.__version__}")
except ImportError:
    print("NumPy 未安装")

try:
    import tensorflow as tf
    print(f"\nTensorFlow 版本: {tf.__version__}")
except ImportError:
    print("TensorFlow 未安装")

try:
    import keras
    print(f"Keras 版本: {keras.__version__}")
except ImportError:
    print("Keras 未安装")

try:
    import sklearn
    print(f"\nScikit-learn 版本: {sklearn.__version__}")
except ImportError:
    print("Scikit-learn 未安装")

try:
    import pandas as pd
    print(f"Pandas 版本: {pd.__version__}")
except ImportError:
    print("Pandas 未安装")

try:
    import matplotlib
    print(f"Matplotlib 版本: {matplotlib.__version__}")
except ImportError:
    print("Matplotlib 未安装")
