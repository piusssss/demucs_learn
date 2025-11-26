#!/usr/bin/env python3
"""
科研绘图脚本：绘制训练损失曲线
支持多个实验对比、论文级别的图表质量
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse

# 设置科研论文风格
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 12
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['grid.alpha'] = 0.3

def load_metrics(exp_dir):
    """从实验目录加载训练指标"""
    exp_path = Path(exp_dir)
    history_file = exp_path / 'history.json'
    
    if not history_file.exists():
        print(f"⚠️  未找到 {history_file}")
        return None
    
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    return history

def smooth_curve(values, weight=0.9):
    """指数移动平均平滑曲线"""
    smoothed = []
    last = values[0] if len(values) > 0 else 0
    for value in values:
        smoothed_val = last * weight + (1 - weight) * value
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def plot_loss_curves(experiments, output_file='training_curves.pdf', 
                     smooth=True, show_valid=True):
    """
    绘制训练损失曲线
    
    Args:
        experiments: dict, {实验名称: 实验目录路径}
        output_file: 输出文件名
        smooth: 是否平滑曲线
        show_valid: 是否显示验证集曲线
    """
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    linestyles = ['-', '--', '-.', ':']
    
    for idx, (name, exp_dir) in enumerate(experiments.items()):
        history = load_metrics(exp_dir)
        if history is None:
            continue
        
        epochs = list(range(1, len(history) + 1))
        
        # 提取训练损失
        train_loss = [h['train']['loss'] for h in history]
        
        # 平滑处理
        if smooth:
            train_loss_smooth = smooth_curve(train_loss, weight=0.9)
            # 绘制平滑曲线
            ax.plot(epochs, train_loss_smooth, color=colors[idx % len(colors)],
                   linestyle=linestyles[0], label=f'{name} (Train)', linewidth=1.5)
        else:
            ax.plot(epochs, train_loss, color=colors[idx % len(colors)],
                   linestyle=linestyles[0], label=f'{name} (Train)', linewidth=1.5)
        
        # 绘制验证损失
        if show_valid:
            valid_data = [(i+1, h['valid']['loss']) for i, h in enumerate(history) if 'valid' in h]
            
            if valid_data:
                valid_epochs, valid_loss = zip(*valid_data)
                valid_epochs = list(valid_epochs)
                valid_loss = list(valid_loss)
                
                if smooth and len(valid_loss) > 1:
                    valid_loss_smooth = smooth_curve(valid_loss, weight=0.8)
                    ax.plot(valid_epochs, valid_loss_smooth, color=colors[idx % len(colors)],
                           linestyle='-', label=f'{name} (Valid)', 
                           linewidth=1.5, alpha=0.7)
                else:
                    ax.plot(valid_epochs, valid_loss, color=colors[idx % len(colors)],
                           linestyle='-', label=f'{name} (Valid)', 
                           linewidth=1.5, alpha=0.7)
    
    # 设置图表样式
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Loss (L1)', fontweight='bold')
    ax.set_title('Training and Validation Loss', fontweight='bold', pad=10)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc='best', framealpha=0.9, edgecolor='gray')
    
    # 设置坐标轴
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ 图表已保存: {output_file}")
    plt.close()

def plot_sdr_curves(experiments, output_file='sdr_curves.pdf'):
    """绘制 SDR 曲线"""
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for idx, (name, exp_dir) in enumerate(experiments.items()):
        history = load_metrics(exp_dir)
        if history is None:
            continue
        
        # 提取有 SDR 的 epoch
        sdr_data = [(i+1, h['valid'].get('nsdr', 0)) 
                    for i, h in enumerate(history) 
                    if 'valid' in h and 'nsdr' in h['valid']]
        
        if not sdr_data:
            continue
        
        epochs, sdrs = zip(*sdr_data)
        
        ax.plot(epochs, sdrs, color=colors[idx % len(colors)],
               marker='o', markersize=4, label=name, linewidth=1.5)
    
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('SDR (dB)', fontweight='bold')
    ax.set_title('Validation SDR over Training', fontweight='bold', pad=10)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc='best', framealpha=0.9, edgecolor='gray')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ 图表已保存: {output_file}")
    plt.close()

def main():
    # 硬编码实验配置
    experiments = {
        'HTDemucs Baseline': 'outputs/xps/60ac4b53',
        'Your Model': 'outputs/xps/e2f418f7',
    }
    
    print(f"📊 绘制实验对比:")
    for name, path in experiments.items():
        if Path(path).exists():
            print(f"  ✅ {name}: {path}")
        else:
            print(f"  ❌ {name}: {path} (不存在)")
    
    # 绘制损失曲线（减少平滑以保持真实性）
    plot_loss_curves(experiments, output_file='training_loss.pdf', smooth=False, show_valid=True)

if __name__ == '__main__':
    main()
