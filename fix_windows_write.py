import os
import time
import contextlib
from pathlib import Path

@contextlib.contextmanager
def windows_safe_write_and_rename(file_path, mode="wb"):
    """
    Windows兼容的文件写入函数，避免文件锁定问题
    兼容Dora的write_and_rename接口
    """
    file_path = Path(file_path)
    temp_path = file_path.with_suffix(file_path.suffix + '.tmp')
    
    # 如果临时文件存在，先删除
    if temp_path.exists():
        try:
            temp_path.unlink()
        except:
            time.sleep(0.1)  # 等待一下再试
            if temp_path.exists():
                temp_path.unlink()
    
    try:
        # 根据模式打开文件
        if mode == "w":
            with open(temp_path, mode, encoding='utf-8') as f:
                yield f
        else:
            with open(temp_path, mode) as f:
                yield f
        
        # 如果目标文件存在，先删除
        if file_path.exists():
            try:
                file_path.unlink()
            except:
                time.sleep(0.1)  # 等待一下再试
                if file_path.exists():
                    file_path.unlink()
        
        # 重命名临时文件
        temp_path.rename(file_path)
    except Exception as e:
        # 清理临时文件
        if temp_path.exists():
            try:
                temp_path.unlink()
            except:
                pass
        raise e

# 替换Dora的write_and_rename函数
import dora.utils
dora.utils.write_and_rename = windows_safe_write_and_rename
