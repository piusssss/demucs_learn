#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试脚本 - 用于在IDE中调试Demucs的separate功能
这个脚本模拟控制台输入，可以直接在IDE中设置断点调试
"""

import sys
import os

# 添加demucs模块到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from demucs.separate import main

if __name__ == "__main__":
    # 模拟命令行参数
    # 相当于运行: demucs -n htdemucs_ft --shifts 2 --two-stems vocals --overlap 0.25 bilibili/war.wav
    debug_args = [
        '-n', 'htdemucs_ft',        # 使用htdemucs_ft模型
        '--shifts', '2',            # 使用2次位移
        '--two-stems', 'vocals',    # 只分离人声轨道
        '--overlap', '0.25',        # 重叠率0.25
        'bilibili/war.wav'          # 输入音频文件
    ]
    
    print(f"开始调试Demucs分离功能...")
    print(f"参数: {' '.join(debug_args)}")
    print("-" * 50)
    
    # 调用main函数，传入调试参数
    # 在这里可以设置断点进行调试
    main(opts=debug_args)
    
    print("-" * 50)
    print("调试完成！")