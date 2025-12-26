#!/usr/bin/env python3
"""检查Transformer LayerScale (gamma) 值"""

import torch
from pathlib import Path
import re

def check_layerscale(model_path):
    print(f"🔍 检查模型: {model_path}")
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['state']
    
    # 查找所有gamma参数
    gamma_keys = [k for k in state_dict.keys() if 'gamma' in k and 'scale' in k]
    
    if not gamma_keys:
        print("❌ 未找到LayerScale参数")
        print("   这个模型可能没有启用LayerScale")
        return
    
    print(f"\n{'='*60}")
    print(f"📊 Transformer LayerScale 释放情况")
    print(f"{'='*60}")
    
    # 分类gamma参数
    # 原版HT: crosstransformer.layers.0.gamma_1.scale, crosstransformer.layers_t.0.gamma_1.scale
    # 2nn: crosstransformer.layers.0.gamma_1.scale, crosstransformer.layers_t.0.gamma_1.scale
    # 2nns: unit_transformers.0.0.layers.0.gamma_1.scale
    
    init_val = 1e-4
    
    # 按transformer组件分组
    transformer_groups = {}
    
    for key in gamma_keys:
        gamma = state_dict[key]
        
        # 解析key的结构
        if 'crosstransformer' in key:
            # 原版HT或2nn: crosstransformer.layers.0.gamma_1.scale
            match = re.search(r'crosstransformer\.(layers|layers_t)\.(\d+)\.(gamma_[12])\.scale', key)
            if match:
                branch = 'Freq' if match.group(1) == 'layers' else 'Time'
                layer_idx = int(match.group(2))
                gamma_type = 'Attn' if match.group(3) == 'gamma_1' else 'FFN'
                group_name = f"crosstransformer.{branch}.L{layer_idx}.{gamma_type}"
                transformer_groups[group_name] = gamma
        
        elif 'unit_transformers' in key:
            # 2nns: unit_transformers.0.0.layers.0.gamma_1.scale
            # 或: unit_transformers.0.1.0.layers.0.gamma_1.scale (有res_idx)
            match = re.search(r'unit_transformers\.(\d+)\.(\d+)(?:\.(\d+))?\..*?layers.*?\.(\d+)\.(gamma_[12])\.scale', key)
            if match:
                t_layer = int(match.group(1))
                step = int(match.group(2))
                res_idx = match.group(3)
                inner_layer = int(match.group(4))
                gamma_type = 'Attn' if match.group(5) == 'gamma_1' else 'FFN'
                
                step_names = ['Step1_Time看Freq', 'Step2_Freq看Time', 'Step3_Freq自注意', 'Step4_Time自注意']
                step_name = step_names[step] if step < len(step_names) else f'Step{step}'
                
                if res_idx:
                    group_name = f"T{t_layer}.{step_name}.Res{res_idx}.L{inner_layer}.{gamma_type}"
                else:
                    group_name = f"T{t_layer}.{step_name}.L{inner_layer}.{gamma_type}"
                
                transformer_groups[group_name] = gamma
    
    if not transformer_groups:
        print("⚠️  找到gamma参数但无法解析结构")
        print(f"   找到的keys: {gamma_keys[:5]}...")
        return
    
    # 统计和显示
    print(f"\n找到 {len(transformer_groups)} 个LayerScale参数\n")
    
    all_means = []
    all_abs_means = []
    all_growth = []
    
    # 按名称顺序（保持层的顺序）
    for name in sorted(transformer_groups.keys()):
        gamma = transformer_groups[name]
        mean_val = gamma.mean().item()
        abs_mean = abs(mean_val)
        growth = mean_val / init_val
        abs_growth = abs(growth)
        
        all_means.append(mean_val)
        all_abs_means.append(abs_mean)
        all_growth.append(growth)
        
        # 方向标记
        direction = "➕" if mean_val >= 0 else "➖"
        
        # 释放程度标记（按绝对值）
        if abs_growth < 2:
            status = "🔒 未释放"
        elif abs_growth < 10:
            status = "🔓 轻微释放"
        elif abs_growth < 100:
            status = "📈 中度释放"
        else:
            status = "🚀 充分释放"
        
        print(f"{direction} {name:48s}: {mean_val:9.6f} ({growth:7.1f}x, |{abs_growth:6.1f}x|) {status}")
    
    # 总体统计
    print(f"\n{'='*60}")
    print(f"📈 总体统计")
    print(f"{'='*60}")
    print(f"平均gamma值: {sum(all_means)/len(all_means):.6f}")
    print(f"平均|gamma|值: {sum(all_abs_means)/len(all_abs_means):.6f}")
    print(f"平均增长倍数: {sum(all_growth)/len(all_growth):.1f}x")
    print(f"平均|增长|倍数: {sum(abs(g) for g in all_growth)/len(all_growth):.1f}x")
    print(f"最小增长: {min(all_growth):.1f}x")
    print(f"最大增长: {max(all_growth):.1f}x")
    
    # 方向统计
    positive = sum(1 for m in all_means if m >= 0)
    negative = sum(1 for m in all_means if m < 0)
    print(f"\n方向分布:")
    print(f"  ➕ 正向增强: {positive:3d} ({positive/len(all_means)*100:.1f}%)")
    print(f"  ➖ 反向抑制: {negative:3d} ({negative/len(all_means)*100:.1f}%)")
    
    # 释放程度分布（按绝对值）
    abs_growth_list = [abs(g) for g in all_growth]
    locked = sum(1 for g in abs_growth_list if g < 2)
    light = sum(1 for g in abs_growth_list if 2 <= g < 10)
    medium = sum(1 for g in abs_growth_list if 10 <= g < 100)
    full = sum(1 for g in abs_growth_list if g >= 100)
    
    print(f"\n释放程度分布（按|gamma|）:")
    print(f"  🔒 未释放 (<2x):    {locked:3d} ({locked/len(abs_growth_list)*100:.1f}%)")
    print(f"  🔓 轻微释放 (2-10x):  {light:3d} ({light/len(abs_growth_list)*100:.1f}%)")
    print(f"  📈 中度释放 (10-100x): {medium:3d} ({medium/len(abs_growth_list)*100:.1f}%)")
    print(f"  🚀 充分释放 (>100x):  {full:3d} ({full/len(abs_growth_list)*100:.1f}%)")
    
    avg_abs_growth = sum(abs_growth_list)/len(abs_growth_list)
    if avg_abs_growth < 10:
        print(f"\n⚠️  Transformer整体释放不足，可能需要更多训练")
    elif avg_abs_growth < 50:
        print(f"\n✓ Transformer正在逐步释放")
    else:
        print(f"\n✅ Transformer已充分释放")

# 硬编码默认模型
default_model = "outputs/xps/htt200/checkpoint.th"

if Path(default_model).exists():
    check_layerscale(default_model)
else:
    print(f"❌ 默认模型不存在: {default_model}")

