import sys

print("正在检查环境配置...")
print(f"1. Python 版本: {sys.version.split()[0]}")

# 检查 PyTorch
try:
    import torch
    print(f"2. PyTorch 版本: {torch.__version__} (成功！)")
    if torch.cuda.is_available():
        print("   -> GPU 加速: 可用 (太棒了！)")
    else:
        print("   -> GPU 加速: 不可用 (目前只能用CPU跑，做轻量级算法没问题)")
except ImportError:
    print("2. [警告] PyTorch 未安装！你需要安装它。")

# 检查 Scipy (读数据用)
try:
    import scipy
    print(f"3. Scipy 版本: {scipy.__version__} (成功！)")
except ImportError:
    print("3. [警告] Scipy 未安装！")

# 检查 Matplotlib (画图用)
try:
    import matplotlib
    print(f"4. Matplotlib 版本: {matplotlib.__version__} (成功！)")
except ImportError:
    print("4. [警告] Matplotlib 未安装！")

# 检查 Pandas (处理表格用)
try:
    import pandas
    print(f"5. Pandas 版本: {pandas.__version__} (成功！)")
except ImportError:
    print("5. [警告] Pandas 未安装！")