#!/usr/bin/env python3
"""
对比两个模型的频谱输出
自动检查separated文件夹，如果没有则运行demucs生成
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
from pathlib import Path
import subprocess
import sys

# ==================== 配置区域 ==================== 
# 在这里硬编码你的配置

# 模型配置
MODEL_HT = "htt100"  # 默认HT模型名称
MODEL_2NN = "e248_100"  # 2nn模型名称

# 音频文件路径（相对于项目根目录）
AUDIO_FILE = "music/back.wav"  # 修改为你的音频文件

# 源名称（要分析的源）
SOURCES = ["drums", "bass", "other", "vocals"]  # 分析所有4个源

# 输出目录
OUTPUT_DIR = "spectrum_analysis"

# Demucs参数
SHIFTS = 1
OVERLAP = 0.25

# ==================== 配置区域结束 ====================


def check_and_generate_separation(model_name, audio_file):
    """
    检查separated文件夹是否有对应音频，如果没有则运行demucs生成
    
    参数:
        model_name: 模型名称
        audio_file: 音频文件路径
    
    返回:
        separated_dir: 分离结果目录
    """
    audio_path = Path(audio_file)
    audio_stem = audio_path.stem  # 文件名（不含扩展名）
    
    # separated文件夹路径
    separated_dir = Path("separated") / model_name / audio_stem
    
    # 检查是否已存在
    if separated_dir.exists():
        # 检查是否有所有源的文件
        sources = ["drums", "bass", "other", "vocals"]
        all_exist = all((separated_dir / f"{source}.wav").exists() for source in sources)
        
        if all_exist:
            print(f"✓ 找到已存在的分离结果: {separated_dir}")
            return separated_dir
        else:
            print(f"⚠ 分离结果不完整，重新生成...")
    
    # 需要生成
    print(f"⚙ 运行demucs生成分离结果...")
    print(f"  模型: {model_name}")
    print(f"  音频: {audio_file}")
    
    # 构建命令
    cmd = [
        "demucs",
        "--repo", "./release_models",
        "-n", model_name,
        f"--shifts={SHIFTS}",
        "--overlap", str(OVERLAP),
        audio_file
    ]
    
    print(f"  命令: {' '.join(cmd)}")
    
    # 运行命令
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✓ 分离完成")
        return separated_dir
    except subprocess.CalledProcessError as e:
        print(f"✗ 分离失败:")
        print(e.stderr)
        sys.exit(1)


def load_source_audio(separated_dir, source_name):
    """
    加载指定源的音频
    
    参数:
        separated_dir: 分离结果目录
        source_name: 源名称
    
    返回:
        audio: 音频数据
        sr: 采样率
    """
    audio_file = separated_dir / f"{source_name}.wav"
    
    if not audio_file.exists():
        raise FileNotFoundError(f"找不到音频文件: {audio_file}")
    
    print(f"  加载: {audio_file}")
    audio, sr = librosa.load(str(audio_file), sr=None, mono=True)
    
    return audio, sr


def plot_spectrum_comparison(audio_ht, audio_2nn, sr, source_name, save_path):
    """
    对比两个模型的频谱
    
    参数:
        audio_ht: HT模型输出
        audio_2nn: 2nn模型输出
        sr: 采样率
        source_name: 源名称
        save_path: 保存路径
    """
    # 计算STFT
    n_fft = 2048
    hop_length = 512
    
    spec_ht = librosa.stft(audio_ht, n_fft=n_fft, hop_length=hop_length)
    spec_2nn = librosa.stft(audio_2nn, n_fft=n_fft, hop_length=hop_length)
    
    # 转换为dB
    spec_ht_db = librosa.amplitude_to_db(np.abs(spec_ht), ref=np.max)
    spec_2nn_db = librosa.amplitude_to_db(np.abs(spec_2nn), ref=np.max)
    
    # 计算差异
    diff = spec_ht_db - spec_2nn_db
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'{source_name.capitalize()} - Model Comparison', 
                 fontsize=16, fontweight='bold')
    
    # 1. HT模型频谱
    img1 = librosa.display.specshow(spec_ht_db, sr=sr, hop_length=hop_length,
                                     x_axis='time', y_axis='hz', ax=axes[0, 0],
                                     cmap='viridis', vmin=-80, vmax=0)
    axes[0, 0].set_title(f'{MODEL_HT}', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Frequency (Hz)')
    axes[0, 0].set_ylim([0, 4000])
    fig.colorbar(img1, ax=axes[0, 0], format='%+2.0f dB')
    
    # 2. 2nn模型频谱
    img2 = librosa.display.specshow(spec_2nn_db, sr=sr, hop_length=hop_length,
                                     x_axis='time', y_axis='hz', ax=axes[0, 1],
                                     cmap='viridis', vmin=-80, vmax=0)
    axes[0, 1].set_title(f'{MODEL_2NN}', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Frequency (Hz)')
    axes[0, 1].set_ylim([0, 4000])
    fig.colorbar(img2, ax=axes[0, 1], format='%+2.0f dB')
    
    # 3. 差异图
    img3 = librosa.display.specshow(diff, sr=sr, hop_length=hop_length,
                                     x_axis='time', y_axis='hz', ax=axes[1, 0],
                                     cmap='RdBu_r', vmin=-20, vmax=20)
    axes[1, 0].set_title('Difference (HT - 2nn)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Frequency (Hz)')
    axes[1, 0].set_ylim([0, 4000])
    fig.colorbar(img3, ax=axes[1, 0], format='%+2.0f dB')
    
    # 4. 频率剖面对比
    mag_ht = np.abs(spec_ht).mean(axis=1)
    mag_2nn = np.abs(spec_2nn).mean(axis=1)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    
    axes[1, 1].plot(freqs, 20*np.log10(mag_ht + 1e-10), 
                   label=MODEL_HT, linewidth=2, alpha=0.8)
    axes[1, 1].plot(freqs, 20*np.log10(mag_2nn + 1e-10), 
                   label=MODEL_2NN, linewidth=2, alpha=0.8)
    axes[1, 1].set_xlim([0, 2000])
    axes[1, 1].set_xlabel('Frequency (Hz)')
    axes[1, 1].set_ylabel('Magnitude (dB)')
    axes[1, 1].set_title('Average Frequency Profile', fontsize=12, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 标注特定频段（根据源类型）
    if source_name.lower() == 'bass':
        for ax in [axes[0, 0], axes[0, 1], axes[1, 0]]:
            ax.axhspan(40, 250, alpha=0.1, color='red')
        axes[1, 1].axvspan(40, 250, alpha=0.2, color='red', label='Bass Range')
    elif source_name.lower() == 'drums':
        for ax in [axes[0, 0], axes[0, 1], axes[1, 0]]:
            ax.axhspan(50, 500, alpha=0.1, color='orange')
        axes[1, 1].axvspan(50, 500, alpha=0.2, color='orange', label='Drums Range')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 频谱对比图已保存: {save_path}")
    plt.close()


def analyze_energy_distribution(audio_ht, audio_2nn, sr, source_name):
    """
    分析两个模型在不同频段的能量分布
    
    参数:
        audio_ht: HT模型输出
        audio_2nn: 2nn模型输出
        sr: 采样率
        source_name: 源名称
    """
    # 计算STFT
    n_fft = 2048
    spec_ht = librosa.stft(audio_ht, n_fft=n_fft)
    spec_2nn = librosa.stft(audio_2nn, n_fft=n_fft)
    
    # 频率轴
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    
    # 定义频段
    bands = {
        'Sub-bass (20-60 Hz)': (20, 60),
        'Bass (60-250 Hz)': (60, 250),
        'Low-mid (250-500 Hz)': (250, 500),
        'Mid (500-2000 Hz)': (500, 2000),
        'High (2000-8000 Hz)': (2000, 8000),
    }
    
    print(f"\n{'='*70}")
    print(f"{source_name.capitalize()} - Energy Distribution Comparison")
    print(f"{'='*70}")
    print(f"{'Frequency Band':<25} {'HT Energy':<15} {'2nn Energy':<15} {'Ratio':<10}")
    print(f"{'-'*70}")
    
    for band_name, (f_low, f_high) in bands.items():
        # 找到频段对应的bin
        bins = np.where((freqs >= f_low) & (freqs <= f_high))[0]
        
        if len(bins) == 0:
            continue
        
        # 计算能量
        energy_ht = np.sum(np.abs(spec_ht[bins, :]) ** 2)
        energy_2nn = np.sum(np.abs(spec_2nn[bins, :]) ** 2)
        
        ratio = energy_ht / energy_2nn if energy_2nn > 0 else float('inf')
        
        print(f"{band_name:<25} {energy_ht:<15.2e} {energy_2nn:<15.2e} {ratio:<10.3f}")
    
    # 总能量
    total_energy_ht = np.sum(np.abs(spec_ht) ** 2)
    total_energy_2nn = np.sum(np.abs(spec_2nn) ** 2)
    total_ratio = total_energy_ht / total_energy_2nn if total_energy_2nn > 0 else float('inf')
    
    print(f"{'-'*70}")
    print(f"{'Total':<25} {total_energy_ht:<15.2e} {total_energy_2nn:<15.2e} {total_ratio:<10.3f}")
    print(f"{'='*70}\n")


def plot_waveform_comparison(audio_ht, audio_2nn, sr, source_name, save_path):
    """
    对比两个模型的波形
    
    参数:
        audio_ht: HT模型输出
        audio_2nn: 2nn模型输出
        sr: 采样率
        source_name: 源名称
        save_path: 保存路径
    """
    # 显示整首歌，但降采样以便可视化
    # 如果音频太长，每隔N个样本取一个点
    max_points = 100000  # 最多显示10万个点
    total_samples = len(audio_ht)
    
    if total_samples > max_points:
        # 降采样
        step = total_samples // max_points
        audio_ht_short = audio_ht[::step]
        audio_2nn_short = audio_2nn[::step]
        duration = total_samples / sr
        time = np.linspace(0, duration, len(audio_ht_short))
    else:
        # 直接显示全部
        audio_ht_short = audio_ht
        audio_2nn_short = audio_2nn
        duration = total_samples / sr
        time = np.linspace(0, duration, total_samples)
    
    # 创建图形
    fig, axes = plt.subplots(3, 1, figsize=(18, 10))
    fig.suptitle(f'{source_name.capitalize()} - Waveform Comparison (Full Track: {duration:.1f}s)', 
                 fontsize=16, fontweight='bold')
    
    # 1. HT模型波形
    axes[0].plot(time, audio_ht_short, linewidth=0.3, alpha=0.8)
    axes[0].set_title(f'{MODEL_HT}', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([0, duration])
    
    # 2. 2nn模型波形
    axes[1].plot(time, audio_2nn_short, linewidth=0.3, alpha=0.8, color='orange')
    axes[1].set_title(f'{MODEL_2NN}', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Amplitude')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim([0, duration])
    
    # 3. 差异
    diff = audio_ht_short - audio_2nn_short
    axes[2].plot(time, diff, linewidth=0.3, alpha=0.8, color='red')
    axes[2].set_title('Difference (HT - 2nn)', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Amplitude')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim([0, duration])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 波形对比图已保存: {save_path}")
    plt.close()


def plot_all_sources_comparison(separated_ht, separated_2nn, audio_stem, output_dir):
    """
    生成所有源的综合对比图
    
    参数:
        separated_ht: HT模型分离结果目录
        separated_2nn: 2nn模型分离结果目录
        audio_stem: 音频文件名（不含扩展名）
        output_dir: 输出目录
    """
    sources = SOURCES
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('All Sources - Frequency Profile Comparison', 
                 fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    
    for idx, source_name in enumerate(sources):
        # 加载音频
        try:
            audio_ht, sr = load_source_audio(separated_ht, source_name)
            audio_2nn, _ = load_source_audio(separated_2nn, source_name)
        except FileNotFoundError:
            continue
        
        # 确保长度一致
        min_len = min(len(audio_ht), len(audio_2nn))
        audio_ht = audio_ht[:min_len]
        audio_2nn = audio_2nn[:min_len]
        
        # 计算STFT
        n_fft = 2048
        spec_ht = librosa.stft(audio_ht, n_fft=n_fft)
        spec_2nn = librosa.stft(audio_2nn, n_fft=n_fft)
        
        # 计算平均频谱
        mag_ht = np.abs(spec_ht).mean(axis=1)
        mag_2nn = np.abs(spec_2nn).mean(axis=1)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        
        # 绘制
        ax = axes[idx]
        ax.plot(freqs, 20*np.log10(mag_ht + 1e-10), 
               label=MODEL_HT, linewidth=2, alpha=0.8)
        ax.plot(freqs, 20*np.log10(mag_2nn + 1e-10), 
               label=MODEL_2NN, linewidth=2, alpha=0.8)
        ax.set_xlim([0, 2000])
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Magnitude (dB)')
        ax.set_title(source_name.capitalize(), fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 标注特定频段
        if source_name == 'bass':
            ax.axvspan(40, 250, alpha=0.2, color='red', label='Bass Range')
        elif source_name == 'drums':
            ax.axvspan(50, 500, alpha=0.2, color='orange', label='Drums Range')
    
    plt.tight_layout()
    
    save_path = output_dir / f"{audio_stem}_all_sources_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 综合对比图已保存: {save_path}")
    plt.close()


def main():
    """
    主函数
    """
    print("="*70)
    print("模型频谱对比分析 - 4源完整分析")
    print("="*70)
    print(f"模型1: {MODEL_HT}")
    print(f"模型2: {MODEL_2NN}")
    print(f"音频: {AUDIO_FILE}")
    print(f"源: {', '.join(SOURCES)}")
    print("="*70)
    
    # 检查音频文件是否存在
    if not Path(AUDIO_FILE).exists():
        print(f"✗ 错误: 找不到音频文件 {AUDIO_FILE}")
        sys.exit(1)
    
    # 创建输出目录
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)
    
    # 1. 检查并生成HT模型的分离结果
    print(f"\n[1/2] 检查 {MODEL_HT} 模型的分离结果...")
    separated_ht = check_and_generate_separation(MODEL_HT, AUDIO_FILE)
    
    # 2. 检查并生成2nn模型的分离结果
    print(f"\n[2/2] 检查 {MODEL_2NN} 模型的分离结果...")
    separated_2nn = check_and_generate_separation(MODEL_2NN, AUDIO_FILE)
    
    audio_stem = Path(AUDIO_FILE).stem
    
    # 3. 对每个源进行分析
    for idx, source_name in enumerate(SOURCES, 1):
        print(f"\n{'='*70}")
        print(f"[{idx}/{len(SOURCES)}] 分析 {source_name.upper()}")
        print(f"{'='*70}")
        
        # 加载音频
        print(f"加载音频...")
        try:
            audio_ht, sr_ht = load_source_audio(separated_ht, source_name)
            audio_2nn, sr_2nn = load_source_audio(separated_2nn, source_name)
        except FileNotFoundError as e:
            print(f"✗ 错误: {e}")
            continue
        
        if sr_ht != sr_2nn:
            print(f"✗ 错误: 采样率不匹配 ({sr_ht} vs {sr_2nn})")
            continue
        
        sr = sr_ht
        print(f"  采样率: {sr} Hz")
        print(f"  HT时长: {len(audio_ht)/sr:.2f} 秒")
        print(f"  2nn时长: {len(audio_2nn)/sr:.2f} 秒")
        
        # 确保长度一致
        min_len = min(len(audio_ht), len(audio_2nn))
        audio_ht = audio_ht[:min_len]
        audio_2nn = audio_2nn[:min_len]
        
        # 生成频谱对比图
        print(f"生成频谱对比图...")
        spectrum_path = output_dir / f"{audio_stem}_{source_name}_spectrum_comparison.png"
        plot_spectrum_comparison(audio_ht, audio_2nn, sr, source_name, spectrum_path)
        
        # 生成波形对比图
        print(f"生成波形对比图...")
        waveform_path = output_dir / f"{audio_stem}_{source_name}_waveform_comparison.png"
        plot_waveform_comparison(audio_ht, audio_2nn, sr, source_name, waveform_path)
        
        # 分析能量分布
        print(f"分析能量分布...")
        analyze_energy_distribution(audio_ht, audio_2nn, sr, source_name)
    
    # 4. 生成综合对比图（所有源的频率剖面）
    print(f"\n{'='*70}")
    print("生成综合对比图...")
    print(f"{'='*70}")
    plot_all_sources_comparison(separated_ht, separated_2nn, audio_stem, output_dir)
    
    print(f"\n{'='*70}")
    print("✓ 全部分析完成！")
    print(f"结果保存在: {output_dir}")
    print(f"  - 每个源的频谱对比图: {len(SOURCES)}张")
    print(f"  - 每个源的波形对比图: {len(SOURCES)}张")
    print(f"  - 综合对比图: 1张")
    print(f"  - 总计: {len(SOURCES)*2 + 1}张图")
    print("="*70)


if __name__ == "__main__":
    main()
