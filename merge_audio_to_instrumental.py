#!/usr/bin/env python3
"""
音频合成脚本
将data/instrumental_separation/test目录中每个子目录的除mixture.wav外的音频文件合成为instrumental.wav
然后删除这些源文件
"""

import os
import glob
import numpy as np
import soundfile as sf
from pathlib import Path
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def merge_audio_files(directory_path):
    """
    删除vocals.wav文件，将bass.wav、drums.wav、other.wav合成为instrumental.wav
    
    Args:
        directory_path (str): 包含音频文件的目录路径
    
    Returns:
        bool: 成功返回True，失败返回False
    """
    try:
        # 首先删除vocals.wav文件（如果存在）
        vocals_path = os.path.join(directory_path, 'vocals.wav')
        if os.path.exists(vocals_path):
            try:
                os.remove(vocals_path)
                logger.info(f"已删除vocals.wav文件: {vocals_path}")
            except Exception as e:
                logger.error(f"删除vocals.wav文件时出错: {e}")
        
        # 指定需要合成的文件
        target_files = ['bass.wav', 'drums.wav', 'other.wav']
        source_files = []
        
        for filename in target_files:
            file_path = os.path.join(directory_path, filename)
            if os.path.exists(file_path):
                source_files.append(file_path)
            else:
                logger.warning(f"文件不存在: {file_path}")
        
        if not source_files:
            logger.warning(f"在目录 {directory_path} 中没有找到需要合成的音频文件 (bass.wav, drums.wav, other.wav)")
            return False
        
        logger.info(f"在目录 {directory_path} 中找到 {len(source_files)} 个音频文件需要合成: {[os.path.basename(f) for f in source_files]}")
        
        # 读取第一个文件获取基本信息
        first_audio, sample_rate = sf.read(source_files[0])
        
        # 如果是单声道，转换为立体声
        if len(first_audio.shape) == 1:
            first_audio = np.column_stack((first_audio, first_audio))
        
        # 初始化合成音频
        merged_audio = first_audio.copy()
        
        # 合成其他音频文件
        for audio_file in source_files[1:]:
            try:
                audio_data, sr = sf.read(audio_file)
                
                # 检查采样率是否一致
                if sr != sample_rate:
                    logger.warning(f"文件 {audio_file} 的采样率 ({sr}) 与第一个文件不一致 ({sample_rate})")
                    continue
                
                # 如果是单声道，转换为立体声
                if len(audio_data.shape) == 1:
                    audio_data = np.column_stack((audio_data, audio_data))
                
                # 确保长度一致（取较短的长度）
                min_length = min(len(merged_audio), len(audio_data))
                merged_audio = merged_audio[:min_length]
                audio_data = audio_data[:min_length]
                
                # 相加合成
                merged_audio = merged_audio + audio_data
                
                logger.info(f"已合成文件: {os.path.basename(audio_file)}")
                
            except Exception as e:
                logger.error(f"处理文件 {audio_file} 时出错: {e}")
                continue
        
        # 保存合成的instrumental.wav
        instrumental_path = os.path.join(directory_path, 'instrumental.wav')
        sf.write(instrumental_path, merged_audio, sample_rate)
        logger.info(f"已保存合成文件: {instrumental_path}")
        
        # 删除源文件
        deleted_files = []
        for source_file in source_files:
            try:
                os.remove(source_file)
                deleted_files.append(os.path.basename(source_file))
                logger.info(f"已删除源文件: {os.path.basename(source_file)}")
            except Exception as e:
                logger.error(f"删除文件 {source_file} 时出错: {e}")
        
        logger.info(f"目录 {directory_path} 处理完成，删除了 {len(deleted_files)} 个源文件")
        return True
        
    except Exception as e:
        logger.error(f"处理目录 {directory_path} 时出错: {e}")
        return False

def main():
    """主函数"""
    # 设置基础路径
    base_path = r"data\instrumental\train"
    
    # 检查基础路径是否存在
    if not os.path.exists(base_path):
        logger.error(f"基础路径不存在: {base_path}")
        return
    
    # 获取所有子目录
    subdirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    
    if not subdirs:
        logger.error(f"在 {base_path} 中没有找到子目录")
        return
    
    logger.info(f"找到 {len(subdirs)} 个子目录需要处理")
    
    # 处理每个子目录
    success_count = 0
    for subdir in subdirs:
        subdir_path = os.path.join(base_path, subdir)
        logger.info(f"正在处理目录: {subdir}")
        
        if merge_audio_files(subdir_path):
            success_count += 1
    
    logger.info(f"处理完成！成功处理了 {success_count}/{len(subdirs)} 个目录")

if __name__ == "__main__":
    main()