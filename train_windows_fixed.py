#!/usr/bin/env python3
"""
Windows兼容的训练启动脚本
解决文件写入冲突问题
硬编码种子参数
"""

import sys
import os

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 应用Windows文件写入修复
from fix_windows_write import windows_safe_write_and_rename
import dora.utils
dora.utils.write_and_rename = windows_safe_write_and_rename

# 应用Windows变调增强修复
print("🔧 应用Windows修复...")
# 修复已经直接在repitch.py中实现，无需额外导入

# 硬编码种子
SEED = 114

# 启动训练
if __name__ == "__main__":
    # 添加硬编码的种子参数
    sys.argv = [sys.argv[0]] + [f"seed={SEED}"] + sys.argv[1:]
    print(f"🎲 使用硬编码随机种子: {SEED}")
    
    from demucs.train import main
    main()
