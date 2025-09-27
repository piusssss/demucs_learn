#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B站视频下载器 - 简化版
直接输入链接下载WAV音频文件到bilibili文件夹
"""

import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """检查必要的依赖是否安装"""
    try:
        import yt_dlp
        print("✅ yt-dlp 已安装")
    except ImportError:
        print("❌ yt-dlp 未安装，正在安装...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "yt-dlp"])
        print("✅ yt-dlp 安装完成")
    
    # 检查ffmpeg
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        print("✅ ffmpeg 已安装")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ ffmpeg 未安装")
        print("请安装ffmpeg:")
        print("Windows: 下载 https://ffmpeg.org/download.html")
        print("macOS: brew install ffmpeg")
        print("Linux: sudo apt install ffmpeg")
        return False
    
    return True

def download_wav(url):
    """下载B站视频并转换为WAV格式"""
    
    # 硬编码输出目录
    output_dir = "bilibili"
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # 配置yt-dlp选项
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': str(output_path / '%(title)s.%(ext)s'),
        'writethumbnail': False,
        'writesubtitles': False,
        'writeautomaticsub': False,
        'ignoreerrors': False,
        'no_warnings': False,
        'quiet': False,
        'verbose': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
    }
    
    try:
        import yt_dlp
        
        print(f"🎵 开始下载: {url}")
        print(f"📁 输出目录: {output_path}")
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # 获取视频信息
            info = ydl.extract_info(url, download=False)
            title = info.get('title', 'unknown')
            print(f"📺 视频标题: {title}")
            
            # 开始下载
            ydl.download([url])
        
        print("✅ 下载完成!")
        
        # 列出下载的文件
        print("\n📋 下载的文件:")
        for file in output_path.glob("*.wav"):
            print(f"  - {file.name}")
        
        return True
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return False

def main():
    """主函数"""
    print("🎵 B站视频下载器 - 简化版")
    print("=" * 50)
    
    # 检查依赖
    if not check_dependencies():
        print("❌ 依赖检查失败，请安装必要的软件")
        return 1
    
    while True:
        print("\n请输入B站视频链接 (输入 'quit' 退出):")
        url = input("链接: ").strip()
        
        if url.lower() in ['quit', 'exit', 'q']:
            print("👋 再见!")
            break
        
        if not url:
            print("❌ 请输入有效的链接")
            continue
        
        if not url.startswith(("http://", "https://")):
            print("❌ 请提供有效的URL")
            continue
        
        # 开始下载
        success = download_wav(url)
        
        if success:
            print("🎉 下载成功!")
        else:
            print("😞 下载失败，请重试")
        
        print("-" * 50)

if __name__ == "__main__":
    sys.exit(main())
