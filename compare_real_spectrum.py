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
import shutil
import json
from datetime import datetime
import pandas as pd

# ==================== 配置区域 ==================== 
# 在这里硬编码你的配置

# 模型配置
MODEL_NAME = "htt100"  # 要对比的模型名称

# MUSDB歌曲目录（包含mixture.wav和各源的.wav文件）
MUSDB_TRACK_DIR = r"data\musdb18_hq_test\test\Carlos Gonzalez - A Place For Us"

# 源名称（要分析的源）
SOURCES = ["drums", "bass", "other", "vocals"]  # 分析所有4个源

# 输出目录（会在此目录下创建模型名子文件夹）
OUTPUT_DIR = "spectrum_analysis_real"

# Demucs参数
SHIFTS = 1
OVERLAP = 0.25

# ==================== 配置区域结束 ====================


def check_and_generate_separation(model_name, musdb_track_dir):
    """
    检查separated文件夹是否有对应音频，如果没有则运行demucs生成
    
    参数:
        model_name: 模型名称
        musdb_track_dir: MUSDB歌曲目录（包含mixture.wav）
    
    返回:
        separated_dir: 分离结果目录
    """
    track_path = Path(musdb_track_dir)
    track_name = track_path.name  # 歌曲名称（目录名）
    mixture_file = track_path / "mixture.wav"
    
    # 检查mixture.wav是否存在
    if not mixture_file.exists():
        print(f"✗ 错误: 找不到 {mixture_file}")
        sys.exit(1)
    
    # separated文件夹路径（使用歌曲名作为文件夹名）
    separated_dir = Path("separated") / model_name / track_name
    
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
    print(f"  音频: {mixture_file}")
    
    # 构建命令
    cmd = [
        "demucs",
        "--repo", "./release_models",
        "-n", model_name,
        f"--shifts={SHIFTS}",
        "--overlap", str(OVERLAP),
        str(mixture_file)
    ]
    
    print(f"  命令: {' '.join(cmd)}")
    
    # 运行命令
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✓ 分离完成")
        
        # demucs会输出到 separated/model_name/mixture/
        # 需要重命名为 separated/model_name/track_name/
        default_output = Path("separated") / model_name / "mixture"
        
        if default_output.exists() and not separated_dir.exists():
            print(f"  重命名输出目录: mixture -> {track_name}")
            import shutil
            shutil.move(str(default_output), str(separated_dir))
        
        return separated_dir
    except subprocess.CalledProcessError as e:
        print(f"✗ 分离失败:")
        print(e.stderr)
        sys.exit(1)


def load_source_audio(source_dir, source_name, is_real=False):
    """
    加载指定源的音频
    
    参数:
        source_dir: 源文件目录
        source_name: 源名称
        is_real: 是否是真实源（MUSDB格式）
    
    返回:
        audio: 音频数据
        sr: 采样率
    """
    if is_real:
        # MUSDB真实源文件名格式
        audio_file = source_dir / f"{source_name}.wav"
    else:
        # 模型输出文件名格式
        audio_file = source_dir / f"{source_name}.wav"
    
    if not audio_file.exists():
        raise FileNotFoundError(f"找不到音频文件: {audio_file}")
    
    print(f"  加载: {audio_file}")
    audio, sr = librosa.load(str(audio_file), sr=None, mono=True)
    
    return audio, sr


def plot_spectrum_comparison(audio_real, audio_model, sr, source_name, save_path):
    """
    对比真实源和模型输出的频谱
    
    参数:
        audio_real: 真实源音频
        audio_model: 模型输出音频
        sr: 采样率
        source_name: 源名称
        save_path: 保存路径
    """
    # 计算STFT
    n_fft = 2048
    hop_length = 512
    
    spec_real = librosa.stft(audio_real, n_fft=n_fft, hop_length=hop_length)
    spec_model = librosa.stft(audio_model, n_fft=n_fft, hop_length=hop_length)
    
    # 转换为dB
    spec_real_db = librosa.amplitude_to_db(np.abs(spec_real), ref=np.max)
    spec_model_db = librosa.amplitude_to_db(np.abs(spec_model), ref=np.max)
    
    # 计算差异
    diff = spec_model_db - spec_real_db
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'{source_name.capitalize()} - Real vs Model Comparison', 
                 fontsize=16, fontweight='bold')
    
    # 1. 真实源频谱
    img1 = librosa.display.specshow(spec_real_db, sr=sr, hop_length=hop_length,
                                     x_axis='time', y_axis='hz', ax=axes[0, 0],
                                     cmap='viridis', vmin=-80, vmax=0)
    axes[0, 0].set_title('Real (Ground Truth)', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Frequency (Hz)')
    axes[0, 0].set_ylim([0, 4000])
    fig.colorbar(img1, ax=axes[0, 0], format='%+2.0f dB')
    
    # 2. 模型输出频谱
    img2 = librosa.display.specshow(spec_model_db, sr=sr, hop_length=hop_length,
                                     x_axis='time', y_axis='hz', ax=axes[0, 1],
                                     cmap='viridis', vmin=-80, vmax=0)
    axes[0, 1].set_title(f'{MODEL_NAME} (Predicted)', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Frequency (Hz)')
    axes[0, 1].set_ylim([0, 4000])
    fig.colorbar(img2, ax=axes[0, 1], format='%+2.0f dB')
    
    # 3. 差异图
    img3 = librosa.display.specshow(diff, sr=sr, hop_length=hop_length,
                                     x_axis='time', y_axis='hz', ax=axes[1, 0],
                                     cmap='RdBu_r', vmin=-20, vmax=20)
    axes[1, 0].set_title('Difference (Model - Real)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Frequency (Hz)')
    axes[1, 0].set_ylim([0, 4000])
    fig.colorbar(img3, ax=axes[1, 0], format='%+2.0f dB')
    
    # 4. 频率剖面对比
    mag_real = np.abs(spec_real).mean(axis=1)
    mag_model = np.abs(spec_model).mean(axis=1)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    
    axes[1, 1].plot(freqs, 20*np.log10(mag_real + 1e-10), 
                   label='Real', linewidth=2, alpha=0.8)
    axes[1, 1].plot(freqs, 20*np.log10(mag_model + 1e-10), 
                   label=MODEL_NAME, linewidth=2, alpha=0.8)
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


def analyze_spectral_similarity(audio_real, audio_model, sr, source_name):
    """
    分析频谱相似度
    
    参数:
        audio_real: 真实源音频
        audio_model: 模型输出音频
        sr: 采样率
        source_name: 源名称
    
    返回:
        similarity_stats: 相似度统计字典
    """
    n_fft = 2048
    spec_real = librosa.stft(audio_real, n_fft=n_fft)
    spec_model = librosa.stft(audio_model, n_fft=n_fft)
    
    # 1. 整体频谱相关系数
    mag_real = np.abs(spec_real).flatten()
    mag_model = np.abs(spec_model).flatten()
    correlation = np.corrcoef(mag_real, mag_model)[0, 1]
    
    # 2. 余弦相似度
    cosine_sim = np.dot(mag_real, mag_model) / (np.linalg.norm(mag_real) * np.linalg.norm(mag_model) + 1e-10)
    
    # 3. 各频段的相关系数
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    bands = {
        'Sub-bass (20-60 Hz)': (20, 60),
        'Bass (60-250 Hz)': (60, 250),
        'Low-mid (250-500 Hz)': (250, 500),
        'Mid (500-2000 Hz)': (500, 2000),
        'High (2000-8000 Hz)': (2000, 8000),
    }
    
    band_correlations = {}
    for band_name, (f_low, f_high) in bands.items():
        bins = np.where((freqs >= f_low) & (freqs <= f_high))[0]
        if len(bins) > 0:
            real_band = np.abs(spec_real[bins, :]).flatten()
            model_band = np.abs(spec_model[bins, :]).flatten()
            if len(real_band) > 1:
                band_corr = np.corrcoef(real_band, model_band)[0, 1]
                band_correlations[band_name] = float(band_corr)
    
    print(f"\n{'='*70}")
    print(f"{source_name.capitalize()} - Spectral Similarity Analysis")
    print(f"{'='*70}")
    print(f"Overall Correlation: {correlation:.4f} (1.0 = perfect match)")
    print(f"Cosine Similarity: {cosine_sim:.4f} (1.0 = perfect match)")
    print(f"\nBand-wise Correlations:")
    for band_name, corr in band_correlations.items():
        print(f"  {band_name:<25} {corr:.4f}")
    print(f"{'='*70}\n")
    
    return {
        'overall_correlation': float(correlation),
        'cosine_similarity': float(cosine_sim),
        'band_correlations': band_correlations
    }


def analyze_error_energy(audio_real, audio_model, sr, source_name):
    """
    分析误差能量（类似SDR的计算）
    
    参数:
        audio_real: 真实源音频
        audio_model: 模型输出音频
        sr: 采样率
        source_name: 源名称
    
    返回:
        error_stats: 误差统计字典
    """
    n_fft = 2048
    spec_real = librosa.stft(audio_real, n_fft=n_fft)
    spec_model = librosa.stft(audio_model, n_fft=n_fft)
    
    # 误差频谱
    error_spec = spec_model - spec_real
    
    # 各频段的误差能量
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    bands = {
        'Sub-bass (20-60 Hz)': (20, 60),
        'Bass (60-250 Hz)': (60, 250),
        'Low-mid (250-500 Hz)': (250, 500),
        'Mid (500-2000 Hz)': (500, 2000),
        'High (2000-8000 Hz)': (2000, 8000),
    }
    
    error_stats = {}
    
    print(f"\n{'='*70}")
    print(f"{source_name.capitalize()} - Error Energy Analysis")
    print(f"{'='*70}")
    print(f"{'Frequency Band':<25} {'Pseudo-SDR':<15} {'Error Ratio':<15}")
    print(f"{'-'*70}")
    
    for band_name, (f_low, f_high) in bands.items():
        bins = np.where((freqs >= f_low) & (freqs <= f_high))[0]
        if len(bins) > 0:
            real_energy = np.sum(np.abs(spec_real[bins, :]) ** 2)
            error_energy = np.sum(np.abs(error_spec[bins, :]) ** 2)
            
            # 类似SDR的计算
            pseudo_sdr = 10 * np.log10(real_energy / (error_energy + 1e-10))
            error_ratio = error_energy / (real_energy + 1e-10)
            
            print(f"{band_name:<25} {pseudo_sdr:<15.3f} {error_ratio:<15.4f}")
            
            error_stats[band_name] = {
                'error_energy': float(error_energy),
                'pseudo_sdr': float(pseudo_sdr),
                'error_ratio': float(error_ratio)
            }
    
    # 总体误差
    total_real_energy = np.sum(np.abs(spec_real) ** 2)
    total_error_energy = np.sum(np.abs(error_spec) ** 2)
    total_pseudo_sdr = 10 * np.log10(total_real_energy / (total_error_energy + 1e-10))
    total_error_ratio = total_error_energy / (total_real_energy + 1e-10)
    
    print(f"{'-'*70}")
    print(f"{'Total':<25} {total_pseudo_sdr:<15.3f} {total_error_ratio:<15.4f}")
    print(f"{'='*70}\n")
    
    error_stats['Total'] = {
        'error_energy': float(total_error_energy),
        'pseudo_sdr': float(total_pseudo_sdr),
        'error_ratio': float(total_error_ratio)
    }
    
    return error_stats


def analyze_silence_leakage(audio_real, audio_model, sr, source_name):
    """
    分析静音段泄漏
    
    参数:
        audio_real: 真实源音频
        audio_model: 模型输出音频
        sr: 采样率
        source_name: 源名称
    
    返回:
        leakage_stats: 泄漏统计字典
    """
    # 计算能量包络
    frame_length = 2048
    hop_length = 512
    
    real_rms = librosa.feature.rms(y=audio_real, frame_length=frame_length, hop_length=hop_length)[0]
    model_rms = librosa.feature.rms(y=audio_model, frame_length=frame_length, hop_length=hop_length)[0]
    
    # 定义静音阈值（真实源RMS最低10%的帧）
    silence_threshold = np.percentile(real_rms, 10)
    
    # 找到静音帧和活跃帧
    silence_frames = real_rms < silence_threshold
    active_frames = ~silence_frames
    
    # 计算静音段和活跃段的输出
    silence_leakage = np.mean(model_rms[silence_frames]) if np.any(silence_frames) else 0
    active_output = np.mean(model_rms[active_frames]) if np.any(active_frames) else 0
    
    leakage_ratio = silence_leakage / active_output if active_output > 0 else 0
    
    # 计算静音段的能量占比
    silence_energy = np.sum(model_rms[silence_frames] ** 2) if np.any(silence_frames) else 0
    total_energy = np.sum(model_rms ** 2)
    silence_energy_ratio = silence_energy / total_energy if total_energy > 0 else 0
    
    print(f"\n{'='*70}")
    print(f"{source_name.capitalize()} - Silence Leakage Analysis")
    print(f"{'='*70}")
    print(f"Silence threshold (RMS): {silence_threshold:.6f}")
    print(f"Silence frames: {np.sum(silence_frames)} / {len(silence_frames)} ({100*np.sum(silence_frames)/len(silence_frames):.1f}%)")
    print(f"Model output in silence: {silence_leakage:.6f} RMS")
    print(f"Model output in active: {active_output:.6f} RMS")
    print(f"Leakage ratio: {leakage_ratio:.4f} (lower is better)")
    print(f"Silence energy ratio: {100*silence_energy_ratio:.2f}% of total output")
    print(f"{'='*70}\n")
    
    return {
        'silence_threshold': float(silence_threshold),
        'silence_frames_count': int(np.sum(silence_frames)),
        'total_frames': int(len(silence_frames)),
        'silence_frames_percentage': float(100 * np.sum(silence_frames) / len(silence_frames)),
        'silence_leakage_rms': float(silence_leakage),
        'active_output_rms': float(active_output),
        'leakage_ratio': float(leakage_ratio),
        'silence_energy_ratio': float(silence_energy_ratio)
    }


def analyze_temporal_alignment(audio_real, audio_model, sr, source_name):
    """
    分析时间对齐和瞬态准确度
    
    参数:
        audio_real: 真实源音频
        audio_model: 模型输出音频
        sr: 采样率
        source_name: 源名称
    
    返回:
        temporal_stats: 时间对齐统计字典
    """
    # 计算onset（瞬态起始点）
    onset_real = librosa.onset.onset_detect(y=audio_real, sr=sr, units='time')
    onset_model = librosa.onset.onset_detect(y=audio_model, sr=sr, units='time')
    
    # 计算onset匹配度
    tolerance = 0.05  # 50ms容差
    matched_onsets = 0
    
    for t_real in onset_real:
        if np.any(np.abs(onset_model - t_real) < tolerance):
            matched_onsets += 1
    
    onset_precision = matched_onsets / len(onset_model) if len(onset_model) > 0 else 0
    onset_recall = matched_onsets / len(onset_real) if len(onset_real) > 0 else 0
    onset_f1 = 2 * onset_precision * onset_recall / (onset_precision + onset_recall) if (onset_precision + onset_recall) > 0 else 0
    
    print(f"\n{'='*70}")
    print(f"{source_name.capitalize()} - Temporal Alignment Analysis")
    print(f"{'='*70}")
    print(f"Real onsets detected: {len(onset_real)}")
    print(f"Model onsets detected: {len(onset_model)}")
    print(f"Matched onsets (±{tolerance*1000:.0f}ms): {matched_onsets}")
    print(f"Onset Precision: {onset_precision:.4f} (how many model onsets are correct)")
    print(f"Onset Recall: {onset_recall:.4f} (how many real onsets are detected)")
    print(f"Onset F1-Score: {onset_f1:.4f}")
    print(f"{'='*70}\n")
    
    return {
        'real_onsets_count': int(len(onset_real)),
        'model_onsets_count': int(len(onset_model)),
        'matched_onsets': int(matched_onsets),
        'onset_precision': float(onset_precision),
        'onset_recall': float(onset_recall),
        'onset_f1': float(onset_f1),
        'tolerance_ms': float(tolerance * 1000)
    }


def analyze_spectral_divergence(audio_real, audio_model, sr, source_name):
    """
    分析频谱分布的差异（KL散度）
    
    参数:
        audio_real: 真实源音频
        audio_model: 模型输出音频
        sr: 采样率
        source_name: 源名称
    
    返回:
        divergence_stats: 散度统计字典
    """
    n_fft = 2048
    spec_real = librosa.stft(audio_real, n_fft=n_fft)
    spec_model = librosa.stft(audio_model, n_fft=n_fft)
    
    # 计算平均频谱（归一化为概率分布）
    mag_real = np.abs(spec_real).mean(axis=1)
    mag_model = np.abs(spec_model).mean(axis=1)
    
    # 归一化
    prob_real = mag_real / (mag_real.sum() + 1e-10)
    prob_model = mag_model / (mag_model.sum() + 1e-10)
    
    # KL散度（Real || Model）
    kl_div = np.sum(prob_real * np.log((prob_real + 1e-10) / (prob_model + 1e-10)))
    
    # JS散度（对称版本，更稳定）
    prob_mean = (prob_real + prob_model) / 2
    js_div = 0.5 * np.sum(prob_real * np.log((prob_real + 1e-10) / (prob_mean + 1e-10))) + \
             0.5 * np.sum(prob_model * np.log((prob_model + 1e-10) / (prob_mean + 1e-10)))
    
    print(f"\n{'='*70}")
    print(f"{source_name.capitalize()} - Spectral Divergence Analysis")
    print(f"{'='*70}")
    print(f"KL Divergence: {kl_div:.6f} (0 = identical, lower is better)")
    print(f"JS Divergence: {js_div:.6f} (0 = identical, lower is better)")
    print(f"{'='*70}\n")
    
    return {
        'kl_divergence': float(kl_div),
        'js_divergence': float(js_div)
    }


def analyze_dynamic_range(audio_real, audio_model, sr, source_name):
    """
    分析动态范围
    
    参数:
        audio_real: 真实源音频
        audio_model: 模型输出音频
        sr: 采样率
        source_name: 源名称
    
    返回:
        dynamic_stats: 动态范围统计字典
    """
    # 计算RMS能量包络
    frame_length = 2048
    hop_length = 512
    
    real_rms = librosa.feature.rms(y=audio_real, frame_length=frame_length, hop_length=hop_length)[0]
    model_rms = librosa.feature.rms(y=audio_model, frame_length=frame_length, hop_length=hop_length)[0]
    
    # 动态范围（dB）
    real_dr = 20 * np.log10(np.max(real_rms) / (np.min(real_rms[real_rms > 0]) + 1e-10))
    model_dr = 20 * np.log10(np.max(model_rms) / (np.min(model_rms[model_rms > 0]) + 1e-10))
    
    # 峰值和RMS
    real_peak = np.max(np.abs(audio_real))
    model_peak = np.max(np.abs(audio_model))
    real_rms_overall = np.sqrt(np.mean(audio_real ** 2))
    model_rms_overall = np.sqrt(np.mean(audio_model ** 2))
    
    # 峰值因子（Crest Factor）
    real_crest = 20 * np.log10(real_peak / (real_rms_overall + 1e-10))
    model_crest = 20 * np.log10(model_peak / (model_rms_overall + 1e-10))
    
    print(f"\n{'='*70}")
    print(f"{source_name.capitalize()} - Dynamic Range Analysis")
    print(f"{'='*70}")
    print(f"{'Metric':<30} {'Real':<15} {'Model':<15} {'Diff':<15}")
    print(f"{'-'*70}")
    print(f"{'Dynamic Range (dB)':<30} {real_dr:<15.2f} {model_dr:<15.2f} {model_dr-real_dr:<15.2f}")
    print(f"{'Peak Amplitude':<30} {real_peak:<15.4f} {model_peak:<15.4f} {model_peak-real_peak:<15.4f}")
    print(f"{'RMS Amplitude':<30} {real_rms_overall:<15.4f} {model_rms_overall:<15.4f} {model_rms_overall-real_rms_overall:<15.4f}")
    print(f"{'Crest Factor (dB)':<30} {real_crest:<15.2f} {model_crest:<15.2f} {model_crest-real_crest:<15.2f}")
    print(f"{'='*70}\n")
    
    return {
        'real_dynamic_range_db': float(real_dr),
        'model_dynamic_range_db': float(model_dr),
        'real_peak': float(real_peak),
        'model_peak': float(model_peak),
        'real_rms': float(real_rms_overall),
        'model_rms': float(model_rms_overall),
        'real_crest_factor_db': float(real_crest),
        'model_crest_factor_db': float(model_crest)
    }


def analyze_energy_distribution(audio_real, audio_model, sr, source_name):
    """
    分析真实源和模型输出在不同频段的能量分布
    
    参数:
        audio_real: 真实源音频
        audio_model: 模型输出音频
        sr: 采样率
        source_name: 源名称
    
    返回:
        energy_stats: 能量统计字典
    """
    # 计算STFT
    n_fft = 2048
    spec_real = librosa.stft(audio_real, n_fft=n_fft)
    spec_model = librosa.stft(audio_model, n_fft=n_fft)
    
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
    
    energy_stats = {}
    
    print(f"\n{'='*70}")
    print(f"{source_name.capitalize()} - Energy Distribution Comparison")
    print(f"{'='*70}")
    print(f"{'Frequency Band':<25} {'Real Energy':<15} {'Model Energy':<15} {'Ratio':<10}")
    print(f"{'-'*70}")
    
    for band_name, (f_low, f_high) in bands.items():
        # 找到频段对应的bin
        bins = np.where((freqs >= f_low) & (freqs <= f_high))[0]
        
        if len(bins) == 0:
            continue
        
        # 计算能量
        energy_real = np.sum(np.abs(spec_real[bins, :]) ** 2)
        energy_model = np.sum(np.abs(spec_model[bins, :]) ** 2)
        
        ratio = energy_model / energy_real if energy_real > 0 else float('inf')
        
        print(f"{band_name:<25} {energy_real:<15.2e} {energy_model:<15.2e} {ratio:<10.3f}")
        
        energy_stats[band_name] = {
            'real_energy': float(energy_real),
            'model_energy': float(energy_model),
            'ratio': float(ratio)
        }
    
    # 总能量
    total_energy_real = np.sum(np.abs(spec_real) ** 2)
    total_energy_model = np.sum(np.abs(spec_model) ** 2)
    total_ratio = total_energy_model / total_energy_real if total_energy_real > 0 else float('inf')
    
    print(f"{'-'*70}")
    print(f"{'Total':<25} {total_energy_real:<15.2e} {total_energy_model:<15.2e} {total_ratio:<10.3f}")
    print(f"{'='*70}\n")
    
    energy_stats['Total'] = {
        'real_energy': float(total_energy_real),
        'model_energy': float(total_energy_model),
        'ratio': float(total_ratio)
    }
    
    return energy_stats


def plot_mask_comparison(audio_real, audio_model, sr, source_name, save_path):
    """
    对比理想掩码和模型掩码
    
    参数:
        audio_real: 真实源音频
        audio_model: 模型输出音频
        sr: 采样率
        source_name: 源名称
        save_path: 保存路径
    """
    n_fft = 2048
    hop_length = 512
    
    spec_real = librosa.stft(audio_real, n_fft=n_fft, hop_length=hop_length)
    spec_model = librosa.stft(audio_model, n_fft=n_fft, hop_length=hop_length)
    
    # 计算掩码（归一化幅度）
    mag_real = np.abs(spec_real)
    mag_model = np.abs(spec_model)
    
    # 归一化到[0, 1]
    mask_real = mag_real / (np.max(mag_real) + 1e-10)
    mask_model = mag_model / (np.max(mag_model) + 1e-10)
    
    # 创建图形（2x2）
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'{source_name.capitalize()} - Mask Comparison', 
                 fontsize=16, fontweight='bold')
    
    # 1. 真实源掩码
    img1 = librosa.display.specshow(mask_real, sr=sr, hop_length=hop_length,
                                     x_axis='time', y_axis='hz', ax=axes[0, 0],
                                     cmap='viridis', vmin=0, vmax=1)
    axes[0, 0].set_title('Ideal Mask (Real)', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Frequency (Hz)')
    axes[0, 0].set_ylim([0, 4000])
    fig.colorbar(img1, ax=axes[0, 0], label='Normalized Magnitude')
    
    # 2. 模型掩码
    img2 = librosa.display.specshow(mask_model, sr=sr, hop_length=hop_length,
                                     x_axis='time', y_axis='hz', ax=axes[0, 1],
                                     cmap='viridis', vmin=0, vmax=1)
    axes[0, 1].set_title('Model Mask', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Frequency (Hz)')
    axes[0, 1].set_ylim([0, 4000])
    fig.colorbar(img2, ax=axes[0, 1], label='Normalized Magnitude')
    
    # 3. 掩码误差
    mask_error = mask_model - mask_real
    img3 = librosa.display.specshow(mask_error, sr=sr, hop_length=hop_length,
                                     x_axis='time', y_axis='hz', ax=axes[1, 0],
                                     cmap='RdBu_r', vmin=-1, vmax=1)
    axes[1, 0].set_title('Mask Error (Model - Real)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Frequency (Hz)')
    axes[1, 0].set_ylim([0, 4000])
    fig.colorbar(img3, ax=axes[1, 0], label='Error')
    
    # 4. 掩码误差直方图
    axes[1, 1].hist(mask_error.flatten(), bins=100, alpha=0.7, edgecolor='black', color='steelblue')
    axes[1, 1].set_xlabel('Mask Error')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Mask Error Distribution', fontsize=12, fontweight='bold')
    axes[1, 1].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    # 添加统计信息
    mean_error = np.mean(mask_error)
    std_error = np.std(mask_error)
    axes[1, 1].text(0.02, 0.98, f'Mean: {mean_error:.4f}\nStd: {std_error:.4f}',
                   transform=axes[1, 1].transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 掩码对比图已保存: {save_path}")
    plt.close()


def plot_energy_envelope(audio_real, audio_model, sr, source_name, save_path):
    """
    对比能量包络
    
    参数:
        audio_real: 真实源音频
        audio_model: 模型输出音频
        sr: 采样率
        source_name: 源名称
        save_path: 保存路径
    """
    frame_length = 2048
    hop_length = 512
    
    real_rms = librosa.feature.rms(y=audio_real, frame_length=frame_length, hop_length=hop_length)[0]
    model_rms = librosa.feature.rms(y=audio_model, frame_length=frame_length, hop_length=hop_length)[0]
    
    times = librosa.frames_to_time(np.arange(len(real_rms)), sr=sr, hop_length=hop_length)
    
    # 显示全曲
    duration = times[-1]
    
    fig, axes = plt.subplots(3, 1, figsize=(18, 10))
    fig.suptitle(f'{source_name.capitalize()} - Energy Envelope Comparison (Full Track: {duration:.1f}s)', 
                 fontsize=16, fontweight='bold')
    
    # 1. 真实能量包络
    axes[0].fill_between(times, 0, real_rms, alpha=0.7, label='Real', color='steelblue')
    axes[0].set_ylabel('RMS Energy')
    axes[0].set_title('Real Energy Envelope', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].set_xlim([0, duration])
    
    # 2. 模型能量包络
    axes[1].fill_between(times, 0, model_rms, alpha=0.7, color='orange', label='Model')
    axes[1].set_ylabel('RMS Energy')
    axes[1].set_title('Model Energy Envelope', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].set_xlim([0, duration])
    
    # 3. 叠加对比
    axes[2].plot(times, real_rms, alpha=0.8, label='Real', linewidth=1.5, color='steelblue')
    axes[2].plot(times, model_rms, alpha=0.8, label='Model', linewidth=1.5, color='orange')
    axes[2].fill_between(times, real_rms, model_rms, alpha=0.3, color='gray', label='Difference')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('RMS Energy')
    axes[2].set_title('Energy Envelope Comparison', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    axes[2].set_xlim([0, duration])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 能量包络图已保存: {save_path}")
    plt.close()


def plot_band_energy_evolution(audio_real, audio_model, sr, source_name, save_path):
    """
    各频段能量随时间变化
    
    参数:
        audio_real: 真实源音频
        audio_model: 模型输出音频
        sr: 采样率
        source_name: 源名称
        save_path: 保存路径
    """
    n_fft = 2048
    hop_length = 512
    
    spec_real = librosa.stft(audio_real, n_fft=n_fft, hop_length=hop_length)
    spec_model = librosa.stft(audio_model, n_fft=n_fft, hop_length=hop_length)
    
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    times = librosa.frames_to_time(np.arange(spec_real.shape[1]), sr=sr, hop_length=hop_length)
    
    # 定义频段
    bands = {
        'Sub-bass (20-60 Hz)': (20, 60),
        'Bass (60-250 Hz)': (60, 250),
        'Low-mid (250-500 Hz)': (250, 500),
        'Mid (500-2000 Hz)': (500, 2000),
        'High (2000-8000 Hz)': (2000, 8000),
    }
    
    # 显示全曲
    duration = times[-1]
    
    fig, axes = plt.subplots(len(bands), 1, figsize=(18, 12))
    fig.suptitle(f'{source_name.capitalize()} - Band Energy Evolution (Full Track: {duration:.1f}s)', 
                 fontsize=16, fontweight='bold')
    
    for idx, (band_name, (f_low, f_high)) in enumerate(bands.items()):
        bins = np.where((freqs >= f_low) & (freqs <= f_high))[0]
        
        if len(bins) > 0:
            # 计算该频段的能量随时间变化
            real_energy = np.sum(np.abs(spec_real[bins, :]) ** 2, axis=0)
            model_energy = np.sum(np.abs(spec_model[bins, :]) ** 2, axis=0)
            
            axes[idx].plot(times, real_energy, alpha=0.8, label='Real', linewidth=1.5, color='steelblue')
            axes[idx].plot(times, model_energy, alpha=0.8, label='Model', linewidth=1.5, color='orange')
            axes[idx].fill_between(times, real_energy, model_energy, alpha=0.2, color='gray')
            axes[idx].set_ylabel('Energy')
            axes[idx].set_title(band_name, fontsize=10, fontweight='bold')
            axes[idx].grid(True, alpha=0.3)
            axes[idx].legend(loc='upper right')
            axes[idx].set_xlim([0, duration])
            
            if idx == len(bands) - 1:
                axes[idx].set_xlabel('Time (s)')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 频段能量演化图已保存: {save_path}")
    plt.close()


def plot_phase_consistency(audio_real, audio_model, sr, source_name, save_path):
    """
    相位一致性分析
    
    参数:
        audio_real: 真实源音频
        audio_model: 模型输出音频
        sr: 采样率
        source_name: 源名称
        save_path: 保存路径
    """
    n_fft = 2048
    hop_length = 512
    
    spec_real = librosa.stft(audio_real, n_fft=n_fft, hop_length=hop_length)
    spec_model = librosa.stft(audio_model, n_fft=n_fft, hop_length=hop_length)
    
    # 计算相位差
    phase_real = np.angle(spec_real)
    phase_model = np.angle(spec_model)
    phase_diff = np.angle(np.exp(1j * (phase_model - phase_real)))  # 归一化到[-π, π]
    
    # 计算相位一致性（余弦）
    phase_consistency = np.cos(phase_diff)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'{source_name.capitalize()} - Phase Consistency Analysis', 
                 fontsize=16, fontweight='bold')
    
    # 1. 真实相位
    img1 = librosa.display.specshow(phase_real, sr=sr, hop_length=hop_length,
                                     x_axis='time', y_axis='hz', ax=axes[0, 0],
                                     cmap='twilight', vmin=-np.pi, vmax=np.pi)
    axes[0, 0].set_title('Real Phase', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Frequency (Hz)')
    axes[0, 0].set_ylim([0, 4000])
    fig.colorbar(img1, ax=axes[0, 0], label='Phase (rad)')
    
    # 2. 模型相位
    img2 = librosa.display.specshow(phase_model, sr=sr, hop_length=hop_length,
                                     x_axis='time', y_axis='hz', ax=axes[0, 1],
                                     cmap='twilight', vmin=-np.pi, vmax=np.pi)
    axes[0, 1].set_title('Model Phase', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Frequency (Hz)')
    axes[0, 1].set_ylim([0, 4000])
    fig.colorbar(img2, ax=axes[0, 1], label='Phase (rad)')
    
    # 3. 相位差
    img3 = librosa.display.specshow(phase_diff, sr=sr, hop_length=hop_length,
                                     x_axis='time', y_axis='hz', ax=axes[1, 0],
                                     cmap='RdBu_r', vmin=-np.pi, vmax=np.pi)
    axes[1, 0].set_title('Phase Difference', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Frequency (Hz)')
    axes[1, 0].set_ylim([0, 4000])
    fig.colorbar(img3, ax=axes[1, 0], label='Phase Diff (rad)')
    
    # 4. 相位一致性
    img4 = librosa.display.specshow(phase_consistency, sr=sr, hop_length=hop_length,
                                     x_axis='time', y_axis='hz', ax=axes[1, 1],
                                     cmap='viridis', vmin=-1, vmax=1)
    axes[1, 1].set_title('Phase Consistency (cos)', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Frequency (Hz)')
    axes[1, 1].set_ylim([0, 4000])
    fig.colorbar(img4, ax=axes[1, 1], label='Consistency')
    
    # 添加统计信息
    mean_consistency = np.mean(phase_consistency)
    axes[1, 1].text(0.02, 0.98, f'Mean Consistency: {mean_consistency:.4f}',
                   transform=axes[1, 1].transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 相位一致性图已保存: {save_path}")
    plt.close()


def plot_waveform_comparison(audio_real, audio_model, sr, source_name, save_path):
    """
    对比真实源和模型输出的波形
    
    参数:
        audio_real: 真实源音频
        audio_model: 模型输出音频
        sr: 采样率
        source_name: 源名称
        save_path: 保存路径
    """
    # 显示整首歌，但降采样以便可视化
    # 如果音频太长，每隔N个样本取一个点
    max_points = 100000  # 最多显示10万个点
    total_samples = len(audio_real)
    
    if total_samples > max_points:
        # 降采样
        step = total_samples // max_points
        audio_real_short = audio_real[::step]
        audio_model_short = audio_model[::step]
        duration = total_samples / sr
        time = np.linspace(0, duration, len(audio_real_short))
    else:
        # 直接显示全部
        audio_real_short = audio_real
        audio_model_short = audio_model
        duration = total_samples / sr
        time = np.linspace(0, duration, total_samples)
    
    # 创建图形
    fig, axes = plt.subplots(3, 1, figsize=(18, 10))
    fig.suptitle(f'{source_name.capitalize()} - Waveform Comparison (Full Track: {duration:.1f}s)', 
                 fontsize=16, fontweight='bold')
    
    # 1. 真实源波形
    axes[0].plot(time, audio_real_short, linewidth=0.3, alpha=0.8)
    axes[0].set_title('Real (Ground Truth)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([0, duration])
    
    # 2. 模型输出波形
    axes[1].plot(time, audio_model_short, linewidth=0.3, alpha=0.8, color='orange')
    axes[1].set_title(f'{MODEL_NAME} (Predicted)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Amplitude')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim([0, duration])
    
    # 3. 差异（误差）
    diff = audio_model_short - audio_real_short
    axes[2].plot(time, diff, linewidth=0.3, alpha=0.8, color='red')
    axes[2].set_title('Error (Model - Real)', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Amplitude')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim([0, duration])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 波形对比图已保存: {save_path}")
    plt.close()


def plot_all_sources_comparison(real_dir, separated_model, track_name, output_dir):
    """
    生成所有源的综合对比图
    
    参数:
        real_dir: 真实源目录
        separated_model: 模型分离结果目录
        track_name: 歌曲名称
        output_dir: 输出目录
    """
    sources = SOURCES
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('All Sources - Frequency Profile Comparison (Real vs Model)', 
                 fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    
    for idx, source_name in enumerate(sources):
        # 加载音频
        try:
            audio_real, sr = load_source_audio(real_dir, source_name, is_real=True)
            audio_model, _ = load_source_audio(separated_model, source_name, is_real=False)
        except FileNotFoundError:
            continue
        
        # 确保长度一致
        min_len = min(len(audio_real), len(audio_model))
        audio_real = audio_real[:min_len]
        audio_model = audio_model[:min_len]
        
        # 计算STFT
        n_fft = 2048
        spec_real = librosa.stft(audio_real, n_fft=n_fft)
        spec_model = librosa.stft(audio_model, n_fft=n_fft)
        
        # 计算平均频谱
        mag_real = np.abs(spec_real).mean(axis=1)
        mag_model = np.abs(spec_model).mean(axis=1)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        
        # 绘制
        ax = axes[idx]
        ax.plot(freqs, 20*np.log10(mag_real + 1e-10), 
               label='Real', linewidth=2, alpha=0.8)
        ax.plot(freqs, 20*np.log10(mag_model + 1e-10), 
               label=MODEL_NAME, linewidth=2, alpha=0.8)
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
    
    save_path = output_dir / "all_sources.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 综合对比图已保存: {save_path}")
    plt.close()


def generate_markdown_report(all_stats, output_dir, track_name):
    """
    生成易读的Markdown报告
    
    参数:
        all_stats: 所有统计数据
        output_dir: 输出目录
        track_name: 歌曲名称
    """
    report_path = output_dir / "report.md"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        # 标题
        f.write(f"# Audio Separation Analysis Report\n\n")
        f.write(f"**Model**: {all_stats['model']}\n\n")
        f.write(f"**Track**: {all_stats['track']}\n\n")
        f.write(f"**Analysis Time**: {all_stats['timestamp']}\n\n")
        f.write("---\n\n")
        
        # 为每个源生成报告
        for source_name in SOURCES:
            if source_name not in all_stats['sources']:
                continue
            
            source_data = all_stats['sources'][source_name]
            
            f.write(f"## {source_name.capitalize()}\n\n")
            f.write(f"**Duration**: {source_data['duration_seconds']:.2f} seconds\n\n")
            f.write(f"**Sample Rate**: {source_data['sample_rate']} Hz\n\n")
            
            # 1. 能量分布
            f.write("### 1. Energy Distribution\n\n")
            f.write("| Frequency Band | Real Energy | Model Energy | Ratio | Status |\n")
            f.write("|---|---|---|---|---|\n")
            
            energy_dist = source_data['energy_distribution']
            for band_name, band_data in energy_dist.items():
                ratio = band_data['ratio']
                if ratio > 1.1:
                    status = "⚠️ Over-extraction"
                elif ratio < 0.9:
                    status = "⚠️ Under-extraction"
                else:
                    status = "✅ Good"
                
                f.write(f"| {band_name} | {band_data['real_energy']:.2e} | {band_data['model_energy']:.2e} | {ratio:.3f} | {status} |\n")
            
            f.write("\n")
            
            # 2. 频谱相似度
            f.write("### 2. Spectral Similarity\n\n")
            similarity = source_data['spectral_similarity']
            f.write(f"- **Overall Correlation**: {similarity['overall_correlation']:.4f} (1.0 = perfect)\n")
            f.write(f"- **Cosine Similarity**: {similarity['cosine_similarity']:.4f} (1.0 = perfect)\n\n")
            
            f.write("**Band-wise Correlations**:\n\n")
            for band_name, corr in similarity['band_correlations'].items():
                status = "✅" if corr > 0.9 else "⚠️" if corr > 0.7 else "❌"
                f.write(f"- {band_name}: {corr:.4f} {status}\n")
            f.write("\n")
            
            # 3. 误差能量（Pseudo-SDR）
            f.write("### 3. Error Energy (Pseudo-SDR)\n\n")
            f.write("| Frequency Band | Pseudo-SDR (dB) | Error Ratio | Quality |\n")
            f.write("|---|---|---|---|\n")
            
            error_energy = source_data['error_energy']
            for band_name, band_data in error_energy.items():
                sdr = band_data['pseudo_sdr']
                if sdr > 10:
                    quality = "✅ Excellent"
                elif sdr > 5:
                    quality = "👍 Good"
                elif sdr > 0:
                    quality = "⚠️ Fair"
                else:
                    quality = "❌ Poor"
                
                f.write(f"| {band_name} | {sdr:.2f} | {band_data['error_ratio']:.4f} | {quality} |\n")
            
            f.write("\n")
            
            # 4. 静音段泄漏
            f.write("### 4. Silence Leakage\n\n")
            leakage = source_data['silence_leakage']
            f.write(f"- **Silence Frames**: {leakage['silence_frames_count']} / {leakage['total_frames']} ({leakage['silence_frames_percentage']:.1f}%)\n")
            f.write(f"- **Leakage Ratio**: {leakage['leakage_ratio']:.4f} (lower is better)\n")
            f.write(f"- **Silence Energy Ratio**: {leakage['silence_energy_ratio']:.4f} ({leakage['silence_energy_ratio']*100:.2f}% of total output)\n")
            
            if leakage['leakage_ratio'] < 0.1:
                f.write(f"- **Status**: ✅ Minimal leakage\n")
            elif leakage['leakage_ratio'] < 0.3:
                f.write(f"- **Status**: ⚠️ Moderate leakage\n")
            else:
                f.write(f"- **Status**: ❌ Significant leakage\n")
            f.write("\n")
            
            # 5. 时间对齐
            f.write("### 5. Temporal Alignment (Onset Detection)\n\n")
            temporal = source_data['temporal_alignment']
            f.write(f"- **Real Onsets**: {temporal['real_onsets_count']}\n")
            f.write(f"- **Model Onsets**: {temporal['model_onsets_count']}\n")
            f.write(f"- **Matched Onsets**: {temporal['matched_onsets']} (±{temporal['tolerance_ms']:.0f}ms)\n")
            f.write(f"- **Precision**: {temporal['onset_precision']:.4f} (how many model onsets are correct)\n")
            f.write(f"- **Recall**: {temporal['onset_recall']:.4f} (how many real onsets are detected)\n")
            f.write(f"- **F1-Score**: {temporal['onset_f1']:.4f}\n")
            
            if temporal['onset_f1'] > 0.8:
                f.write(f"- **Status**: ✅ Excellent temporal alignment\n")
            elif temporal['onset_f1'] > 0.6:
                f.write(f"- **Status**: 👍 Good temporal alignment\n")
            else:
                f.write(f"- **Status**: ⚠️ Poor temporal alignment\n")
            f.write("\n")
            
            # 6. 频谱散度
            f.write("### 6. Spectral Divergence\n\n")
            divergence = source_data['spectral_divergence']
            f.write(f"- **KL Divergence**: {divergence['kl_divergence']:.6f} (0 = identical)\n")
            f.write(f"- **JS Divergence**: {divergence['js_divergence']:.6f} (0 = identical)\n\n")
            
            # 7. 动态范围
            f.write("### 7. Dynamic Range\n\n")
            f.write("| Metric | Real | Model | Difference |\n")
            f.write("|---|---|---|---|\n")
            
            dynamic = source_data['dynamic_range']
            f.write(f"| Dynamic Range (dB) | {dynamic['real_dynamic_range_db']:.2f} | {dynamic['model_dynamic_range_db']:.2f} | {dynamic['model_dynamic_range_db']-dynamic['real_dynamic_range_db']:.2f} |\n")
            f.write(f"| Peak Amplitude | {dynamic['real_peak']:.4f} | {dynamic['model_peak']:.4f} | {dynamic['model_peak']-dynamic['real_peak']:.4f} |\n")
            f.write(f"| RMS Amplitude | {dynamic['real_rms']:.4f} | {dynamic['model_rms']:.4f} | {dynamic['model_rms']-dynamic['real_rms']:.4f} |\n")
            f.write(f"| Crest Factor (dB) | {dynamic['real_crest_factor_db']:.2f} | {dynamic['model_crest_factor_db']:.2f} | {dynamic['model_crest_factor_db']-dynamic['real_crest_factor_db']:.2f} |\n")
            
            f.write("\n---\n\n")
        
        # 总结
        f.write("## Summary\n\n")
        f.write("### Overall Assessment\n\n")
        
        # 计算各源的总体评分
        for source_name in SOURCES:
            if source_name not in all_stats['sources']:
                continue
            
            source_data = all_stats['sources'][source_name]
            
            # 简单评分系统
            score = 0
            total = 0
            
            # 能量比接近1
            energy_ratio = source_data['energy_distribution']['Total']['ratio']
            if 0.9 <= energy_ratio <= 1.1:
                score += 2
            elif 0.8 <= energy_ratio <= 1.2:
                score += 1
            total += 2
            
            # 相关系数高
            corr = source_data['spectral_similarity']['overall_correlation']
            if corr > 0.9:
                score += 2
            elif corr > 0.7:
                score += 1
            total += 2
            
            # Pseudo-SDR高
            pseudo_sdr = source_data['error_energy']['Total']['pseudo_sdr']
            if pseudo_sdr > 10:
                score += 2
            elif pseudo_sdr > 5:
                score += 1
            total += 2
            
            # 泄漏低
            leakage_ratio = source_data['silence_leakage']['leakage_ratio']
            if leakage_ratio < 0.1:
                score += 2
            elif leakage_ratio < 0.3:
                score += 1
            total += 2
            
            # Onset F1高
            onset_f1 = source_data['temporal_alignment']['onset_f1']
            if onset_f1 > 0.8:
                score += 2
            elif onset_f1 > 0.6:
                score += 1
            total += 2
            
            percentage = (score / total) * 100
            
            if percentage >= 80:
                rating = "✅ Excellent"
            elif percentage >= 60:
                rating = "👍 Good"
            elif percentage >= 40:
                rating = "⚠️ Fair"
            else:
                rating = "❌ Poor"
            
            f.write(f"- **{source_name.capitalize()}**: {percentage:.0f}% {rating}\n")
        
        f.write("\n")
        f.write("### Generated Visualizations\n\n")
        f.write(f"- Spectrum comparison: {len(SOURCES)} images\n")
        f.write(f"- Waveform comparison: {len(SOURCES)} images\n")
        f.write(f"- Mask comparison: {len(SOURCES)} images\n")
        f.write(f"- Energy envelope: {len(SOURCES)} images\n")
        f.write(f"- Band energy evolution: {len(SOURCES)} images\n")
        f.write(f"- Phase consistency: {len(SOURCES)} images\n")
        f.write(f"- Comprehensive comparison: 1 image\n")
        f.write(f"\n**Total**: {len(SOURCES)*6 + 1} images\n")
    
    print(f"✓ Markdown报告已保存: {report_path}")
    return report_path


def generate_html_report(all_stats, output_dir, track_name):
    """
    生成HTML可视化报告
    
    参数:
        all_stats: 所有统计数据
        output_dir: 输出目录
        track_name: 歌曲名称
    """
    report_path = output_dir / "report.html"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        # HTML头部
        f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Audio Separation Analysis Report</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }
        .header h1 {
            margin: 0 0 10px 0;
        }
        .header p {
            margin: 5px 0;
            opacity: 0.9;
        }
        .source-section {
            background: white;
            padding: 25px;
            margin-bottom: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .source-section h2 {
            color: #667eea;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
            margin-top: 0;
        }
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .metric-card {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }
        .metric-card h4 {
            margin: 0 0 10px 0;
            color: #333;
            font-size: 14px;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #667eea;
        }
        .metric-label {
            font-size: 12px;
            color: #666;
            margin-top: 5px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #667eea;
            color: white;
            font-weight: 600;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
        .status-excellent { color: #28a745; font-weight: bold; }
        .status-good { color: #17a2b8; font-weight: bold; }
        .status-fair { color: #ffc107; font-weight: bold; }
        .status-poor { color: #dc3545; font-weight: bold; }
        .summary {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 25px;
            border-radius: 10px;
            margin-top: 30px;
        }
        .summary h2 {
            margin-top: 0;
        }
        .score-bar {
            background: rgba(255,255,255,0.3);
            height: 30px;
            border-radius: 15px;
            overflow: hidden;
            margin: 10px 0;
        }
        .score-fill {
            background: white;
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            transition: width 0.5s ease;
        }
    </style>
</head>
<body>
""")
        
        # 标题部分
        f.write(f"""
    <div class="header">
        <h1>🎵 Audio Separation Analysis Report</h1>
        <p><strong>Model:</strong> {all_stats['model']}</p>
        <p><strong>Track:</strong> {all_stats['track']}</p>
        <p><strong>Analysis Time:</strong> {all_stats['timestamp']}</p>
    </div>
""")
        
        # 为每个源生成报告
        for source_name in SOURCES:
            if source_name not in all_stats['sources']:
                continue
            
            source_data = all_stats['sources'][source_name]
            
            # 源标题
            emoji_map = {'drums': '🥁', 'bass': '🎸', 'other': '🎹', 'vocals': '🎤'}
            emoji = emoji_map.get(source_name, '🎵')
            
            f.write(f"""
    <div class="source-section">
        <h2>{emoji} {source_name.capitalize()}</h2>
        <p><strong>Duration:</strong> {source_data['duration_seconds']:.2f}s | <strong>Sample Rate:</strong> {source_data['sample_rate']} Hz</p>
""")
            
            # 关键指标卡片
            similarity = source_data['spectral_similarity']
            error_energy = source_data['error_energy']
            leakage = source_data['silence_leakage']
            temporal = source_data['temporal_alignment']
            
            f.write("""
        <div class="metric-grid">
""")
            
            f.write(f"""
            <div class="metric-card">
                <h4>Spectral Correlation</h4>
                <div class="metric-value">{similarity['overall_correlation']:.3f}</div>
                <div class="metric-label">1.0 = perfect match</div>
            </div>
            <div class="metric-card">
                <h4>Pseudo-SDR</h4>
                <div class="metric-value">{error_energy['Total']['pseudo_sdr']:.2f} dB</div>
                <div class="metric-label">Higher is better</div>
            </div>
            <div class="metric-card">
                <h4>Silence Leakage</h4>
                <div class="metric-value">{leakage['leakage_ratio']:.3f}</div>
                <div class="metric-label">Lower is better</div>
            </div>
            <div class="metric-card">
                <h4>Onset F1-Score</h4>
                <div class="metric-value">{temporal['onset_f1']:.3f}</div>
                <div class="metric-label">Temporal accuracy</div>
            </div>
""")
            
            f.write("""
        </div>
""")
            
            # 能量分布表格
            f.write("""
        <h3>Energy Distribution</h3>
        <table>
            <tr>
                <th>Frequency Band</th>
                <th>Real Energy</th>
                <th>Model Energy</th>
                <th>Ratio</th>
                <th>Status</th>
            </tr>
""")
            
            energy_dist = source_data['energy_distribution']
            for band_name, band_data in energy_dist.items():
                ratio = band_data['ratio']
                if ratio > 1.1:
                    status = '<span class="status-fair">⚠️ Over</span>'
                elif ratio < 0.9:
                    status = '<span class="status-fair">⚠️ Under</span>'
                else:
                    status = '<span class="status-excellent">✅ Good</span>'
                
                f.write(f"""
            <tr>
                <td>{band_name}</td>
                <td>{band_data['real_energy']:.2e}</td>
                <td>{band_data['model_energy']:.2e}</td>
                <td>{ratio:.3f}</td>
                <td>{status}</td>
            </tr>
""")
            
            f.write("""
        </table>
""")
            
            f.write("""
    </div>
""")
        
        # 总结部分
        f.write("""
    <div class="summary">
        <h2>📊 Overall Assessment</h2>
""")
        
        for source_name in SOURCES:
            if source_name not in all_stats['sources']:
                continue
            
            source_data = all_stats['sources'][source_name]
            
            # 计算评分
            score = 0
            total = 10
            
            energy_ratio = source_data['energy_distribution']['Total']['ratio']
            if 0.9 <= energy_ratio <= 1.1:
                score += 2
            elif 0.8 <= energy_ratio <= 1.2:
                score += 1
            
            corr = source_data['spectral_similarity']['overall_correlation']
            if corr > 0.9:
                score += 2
            elif corr > 0.7:
                score += 1
            
            pseudo_sdr = source_data['error_energy']['Total']['pseudo_sdr']
            if pseudo_sdr > 10:
                score += 2
            elif pseudo_sdr > 5:
                score += 1
            
            leakage_ratio = source_data['silence_leakage']['leakage_ratio']
            if leakage_ratio < 0.1:
                score += 2
            elif leakage_ratio < 0.3:
                score += 1
            
            onset_f1 = source_data['temporal_alignment']['onset_f1']
            if onset_f1 > 0.8:
                score += 2
            elif onset_f1 > 0.6:
                score += 1
            
            percentage = (score / total) * 100
            
            if percentage >= 80:
                rating = "✅ Excellent"
            elif percentage >= 60:
                rating = "👍 Good"
            elif percentage >= 40:
                rating = "⚠️ Fair"
            else:
                rating = "❌ Poor"
            
            f.write(f"""
        <h3>{source_name.capitalize()}: {rating}</h3>
        <div class="score-bar">
            <div class="score-fill" style="width: {percentage}%;">{percentage:.0f}%</div>
        </div>
""")
        
        f.write("""
    </div>
""")
        
        # HTML尾部
        f.write("""
</body>
</html>
""")
    
    print(f"✓ HTML报告已保存: {report_path}")
    return report_path


def main():
    """
    主函数
    """
    print("="*70)
    print("模型 vs 真实源 频谱对比分析")
    print("="*70)
    print(f"模型: {MODEL_NAME}")
    print(f"MUSDB歌曲目录: {MUSDB_TRACK_DIR}")
    print(f"源: {', '.join(SOURCES)}")
    print("="*70)
    
    # 检查MUSDB目录是否存在
    track_dir = Path(MUSDB_TRACK_DIR)
    if not track_dir.exists():
        print(f"✗ 错误: 找不到MUSDB歌曲目录 {MUSDB_TRACK_DIR}")
        sys.exit(1)
    
    track_name = track_dir.name
    
    # 创建输出目录（包含模型名子文件夹）
    output_dir = Path(OUTPUT_DIR) / MODEL_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 检查并生成模型的分离结果
    print(f"\n[1/1] 检查 {MODEL_NAME} 模型的分离结果...")
    separated_model = check_and_generate_separation(MODEL_NAME, track_dir)
    
    # 用于保存所有数值结果
    all_stats = {
        'model': MODEL_NAME,
        'track': track_name,
        'timestamp': datetime.now().isoformat(),
        'sources': {}
    }
    
    # 2. 对每个源进行分析
    for idx, source_name in enumerate(SOURCES, 1):
        print(f"\n{'='*70}")
        print(f"[{idx}/{len(SOURCES)}] 分析 {source_name.upper()}")
        print(f"{'='*70}")
        
        # 加载音频
        print(f"加载音频...")
        try:
            audio_real, sr_real = load_source_audio(track_dir, source_name, is_real=True)
            audio_model, sr_model = load_source_audio(separated_model, source_name, is_real=False)
        except FileNotFoundError as e:
            print(f"✗ 错误: {e}")
            continue
        
        if sr_real != sr_model:
            print(f"✗ 错误: 采样率不匹配 ({sr_real} vs {sr_model})")
            continue
        
        sr = sr_real
        print(f"  采样率: {sr} Hz")
        print(f"  Real时长: {len(audio_real)/sr:.2f} 秒")
        print(f"  Model时长: {len(audio_model)/sr:.2f} 秒")
        
        # 确保长度一致
        min_len = min(len(audio_real), len(audio_model))
        audio_real = audio_real[:min_len]
        audio_model = audio_model[:min_len]
        
        # 生成频谱对比图
        print(f"生成频谱对比图...")
        spectrum_path = output_dir / f"{source_name}_spectrum.png"
        plot_spectrum_comparison(audio_real, audio_model, sr, source_name, spectrum_path)
        
        # 生成波形对比图
        print(f"生成波形对比图...")
        waveform_path = output_dir / f"{source_name}_waveform.png"
        plot_waveform_comparison(audio_real, audio_model, sr, source_name, waveform_path)
        
        # 生成掩码对比图
        print(f"生成掩码对比图...")
        mask_path = output_dir / f"{source_name}_mask.png"
        plot_mask_comparison(audio_real, audio_model, sr, source_name, mask_path)
        
        # 生成能量包络图
        print(f"生成能量包络图...")
        envelope_path = output_dir / f"{source_name}_envelope.png"
        plot_energy_envelope(audio_real, audio_model, sr, source_name, envelope_path)
        
        # 生成频段能量演化图
        print(f"生成频段能量演化图...")
        band_evolution_path = output_dir / f"{source_name}_band_evolution.png"
        plot_band_energy_evolution(audio_real, audio_model, sr, source_name, band_evolution_path)
        
        # 生成相位一致性图
        print(f"生成相位一致性图...")
        phase_path = output_dir / f"{source_name}_phase.png"
        plot_phase_consistency(audio_real, audio_model, sr, source_name, phase_path)
        
        # 分析能量分布
        print(f"分析能量分布...")
        energy_stats = analyze_energy_distribution(audio_real, audio_model, sr, source_name)
        
        # 分析频谱相似度
        print(f"分析频谱相似度...")
        similarity_stats = analyze_spectral_similarity(audio_real, audio_model, sr, source_name)
        
        # 分析误差能量
        print(f"分析误差能量...")
        error_stats = analyze_error_energy(audio_real, audio_model, sr, source_name)
        
        # 分析静音段泄漏
        print(f"分析静音段泄漏...")
        leakage_stats = analyze_silence_leakage(audio_real, audio_model, sr, source_name)
        
        # 分析时间对齐
        print(f"分析时间对齐...")
        temporal_stats = analyze_temporal_alignment(audio_real, audio_model, sr, source_name)
        
        # 分析频谱散度
        print(f"分析频谱散度...")
        divergence_stats = analyze_spectral_divergence(audio_real, audio_model, sr, source_name)
        
        # 分析动态范围
        print(f"分析动态范围...")
        dynamic_stats = analyze_dynamic_range(audio_real, audio_model, sr, source_name)
        
        # 保存到统计字典
        all_stats['sources'][source_name] = {
            'sample_rate': sr,
            'duration_seconds': float(min_len / sr),
            'energy_distribution': energy_stats,
            'spectral_similarity': similarity_stats,
            'error_energy': error_stats,
            'silence_leakage': leakage_stats,
            'temporal_alignment': temporal_stats,
            'spectral_divergence': divergence_stats,
            'dynamic_range': dynamic_stats
        }
    
    # 3. 生成综合对比图（所有源的频率剖面）
    print(f"\n{'='*70}")
    print("生成综合对比图...")
    print(f"{'='*70}")
    plot_all_sources_comparison(track_dir, separated_model, track_name, output_dir)
    
    # 4. 保存数值结果到JSON文件
    stats_file = output_dir / "analysis_data.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)
    print(f"✓ 能量统计数据已保存: {stats_file}")
    
    # 5. 生成Markdown报告
    print(f"\n{'='*70}")
    print("生成易读报告...")
    print(f"{'='*70}")
    markdown_report = generate_markdown_report(all_stats, output_dir, track_name)
    
    # 6. 生成HTML报告
    html_report = generate_html_report(all_stats, output_dir, track_name)
    
    print(f"\n{'='*70}")
    print("✓ 全部分析完成！")
    print(f"结果保存在: {output_dir}")
    print(f"\n生成的图表:")
    print(f"  - 频谱对比图: {len(SOURCES)}张")
    print(f"  - 波形对比图: {len(SOURCES)}张")
    print(f"  - 掩码对比图: {len(SOURCES)}张 (NEW)")
    print(f"  - 能量包络图: {len(SOURCES)}张 (NEW)")
    print(f"  - 频段能量演化图: {len(SOURCES)}张 (NEW)")
    print(f"  - 相位一致性图: {len(SOURCES)}张 (NEW)")
    print(f"  - 综合对比图: 1张")
    print(f"\n生成的报告:")
    print(f"  - JSON数据文件: 1个 (机器可读)")
    print(f"  - Markdown报告: 1个 (易读文本)")
    print(f"  - HTML报告: 1个 (可视化网页)")
    print(f"\n分析指标包括:")
    print(f"  1. 能量分布 (各频段能量对比)")
    print(f"  2. 频谱相似度 (相关系数、余弦相似度)")
    print(f"  3. 误差能量 (Pseudo-SDR)")
    print(f"  4. 静音段泄漏 (泄漏比例)")
    print(f"  5. 时间对齐 (Onset检测)")
    print(f"  6. 频谱散度 (KL/JS散度)")
    print(f"  7. 动态范围 (峰值、RMS、Crest Factor)")
    print(f"\n  总计: {len(SOURCES)*6 + 1}张图 + 3个报告文件 + 7类分析指标")
    print("="*70)


if __name__ == "__main__":
    main()
