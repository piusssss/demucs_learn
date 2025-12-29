#!/usr/bin/env python3
"""检查Transformer LayerScale (gamma) 值"""

import torch
from pathlib import Path
import re
import sys

def check_layerscale(model_path, is_reference=False):
    print(f"🔍 检查模型: {model_path}")
    if is_reference:
        print("   (作为参考标准)")
    
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
    # NF: stft_transformers.0.0.layers.0.gamma_1.scale (layer_idx.band_idx.layers.inner_layer)
    
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
        
        elif 'stft_transformers' in key:
            # NF: stft_transformers.0.0.layers.0.gamma_1.scale
            # layer_idx: encoder层索引 (0-3)
            # band_idx: 频带索引 (Layer0: 0-31, Layer1: 0-11, Layer2: 0-3, Layer3: 0)
            # inner_layer: transformer内部层索引
            match = re.search(r'stft_transformers\.(\d+)\.(\d+)\.layers\.(\d+)\.(gamma_[12])\.scale', key)
            if match:
                layer_idx = int(match.group(1))
                band_idx = int(match.group(2))
                inner_layer = int(match.group(3))
                gamma_type = 'Attn' if match.group(4) == 'gamma_1' else 'FFN'
                
                # 频带数量：Layer0=32, Layer1=12, Layer2=4, Layer3=1
                bands_per_layer = [32, 12, 4, 1]
                total_bands = bands_per_layer[layer_idx] if layer_idx < len(bands_per_layer) else 1
                
                group_name = f"NF.Layer{layer_idx}.Band{band_idx:02d}/{total_bands:02d}.L{inner_layer}.{gamma_type}"
                transformer_groups[group_name] = gamma
        
        elif 'unit_transformers' in key:
            # 2nns有两种transformer：
            # 1. SingleCrossTransformerEncoder (step 0,1,3,4,5): 有res_idx
            #    格式: unit_transformers.{t_layer}.{step}.{res_idx}.layers.{inner_layer}.gamma_{1/2}.scale
            # 2. ConcatCrossTransformerEncoder (step 2): 无res_idx，但有layers/layers_t
            #    格式: unit_transformers.{t_layer}.{step}.layers.{inner_layer}.gamma_{1/2}.scale
            #    或: unit_transformers.{t_layer}.{step}.layers_t.{inner_layer}.gamma_{1/2}.scale
            
            # 尝试匹配有res_idx的情况 (SingleCrossTransformerEncoder)
            match = re.search(r'unit_transformers\.(\d+)\.(\d+)\.(\d+)\.layers\.(\d+)\.(gamma_[12])\.scale', key)
            if match:
                t_layer = int(match.group(1))
                step = int(match.group(2))
                res_idx = int(match.group(3))
                inner_layer = int(match.group(4))
                gamma_type = 'Attn' if match.group(5) == 'gamma_1' else 'FFN'
                
                step_names = {
                    0: 'Freq自注意',
                    1: 'Time自注意',
                    2: 'Time看Freq',
                    3: 'Freq看Time',
                    4: 'Freq自注意2',
                    5: 'Time自注意2'
                }
                step_name = step_names.get(step, f'Step{step}')
                
                group_name = f"T{t_layer}.{step_name}.Res{res_idx}.L{inner_layer}.{gamma_type}"
                transformer_groups[group_name] = gamma
            else:
                # 尝试匹配没有res_idx的情况 (ConcatCrossTransformerEncoder或Time分支)
                # 可能是 layers 或 layers_t
                match = re.search(r'unit_transformers\.(\d+)\.(\d+)\.(layers|layers_t)\.(\d+)\.(gamma_[12])\.scale', key)
                if match:
                    t_layer = int(match.group(1))
                    step = int(match.group(2))
                    branch = match.group(3)  # 'layers' or 'layers_t'
                    inner_layer = int(match.group(4))
                    gamma_type = 'Attn' if match.group(5) == 'gamma_1' else 'FFN'
                    
                    step_names = {
                        0: 'Freq自注意',
                        1: 'Time自注意',
                        2: 'Time看Freq',
                        3: 'Freq看Time',
                        4: 'Freq自注意2',
                        5: 'Time自注意2'
                    }
                    step_name = step_names.get(step, f'Step{step}')
                    
                    # 如果是step 2 (Time看Freq)，区分Freq和Time分支
                    if step == 2:
                        if branch == 'layers':
                            branch_name = 'Freq分支'
                        else:  # layers_t
                            branch_name = 'Time分支'
                        group_name = f"T{t_layer}.{step_name}.{branch_name}.L{inner_layer}.{gamma_type}"
                    else:
                        group_name = f"T{t_layer}.{step_name}.L{inner_layer}.{gamma_type}"
                    
                    transformer_groups[group_name] = gamma
    
    if not transformer_groups:
        print("⚠️  找到gamma参数但无法解析结构")
        print(f"   找到的keys: {gamma_keys[:5]}...")
        return
    
    # 统计和显示
    print(f"\n找到 {len(transformer_groups)} 个LayerScale参数\n")
    
    # 检测是否是NF模型或2nns模型
    is_nf_model = any('stft_transformers' in name for name in transformer_groups.keys())
    is_2nns_model = any('unit_transformers' in name for name in transformer_groups.keys())
    
    all_means = []
    all_abs_means = []
    all_growth = []
    
    # 如果是NF模型，按层分组统计
    if is_nf_model:
        print("🎯 NF模型：按层和频带显示LayerScale\n")
        
        # 按层分组
        layer_groups = {}
        for name in sorted(transformer_groups.keys()):
            if 'NF.Layer' in name:
                layer_match = re.search(r'NF\.Layer(\d+)', name)
                if layer_match:
                    layer_idx = int(layer_match.group(1))
                    if layer_idx not in layer_groups:
                        layer_groups[layer_idx] = []
                    layer_groups[layer_idx].append(name)
        
        # 按层显示
        for layer_idx in sorted(layer_groups.keys()):
            layer_names = layer_groups[layer_idx]
            channels = 48 * (2 ** layer_idx)
            bands_count = len(set(re.search(r'Band(\d+)', n).group(1) for n in layer_names if 'Band' in n))
            
            print(f"{'='*60}")
            print(f"Layer {layer_idx} ({channels} channels, {bands_count} bands)")
            print(f"{'='*60}")
            
            layer_means = []
            for name in sorted(layer_names):
                gamma = transformer_groups[name]
                mean_val = gamma.mean().item()
                abs_mean = abs(mean_val)
                growth = mean_val / init_val
                abs_growth = abs(growth)
                
                all_means.append(mean_val)
                all_abs_means.append(abs_mean)
                all_growth.append(growth)
                layer_means.append(abs_mean)
                
                # 方向标记
                direction = "➕" if mean_val >= 0 else "➖"
                
                # 释放程度标记（按绝对值）
                if abs_growth < 2:
                    status = "🔒"
                elif abs_growth < 10:
                    status = "🔓"
                elif abs_growth < 100:
                    status = "📈"
                else:
                    status = "🚀"
                
                # 简化显示：只显示Band和类型
                short_name = re.sub(r'NF\.Layer\d+\.', '', name)
                print(f"{direction} {short_name:35s}: {mean_val:9.6f} ({growth:7.1f}x, |{abs_growth:6.1f}x|) {status}")
            
            # 层统计
            layer_avg = sum(layer_means) / len(layer_means)
            print(f"\n  Layer {layer_idx} 平均|gamma|: {layer_avg:.6f} ({layer_avg/init_val:.1f}x)")
            print()
    
    elif is_2nns_model:
        print("🎯 2nns模型：按Transformer层和Step显示LayerScale\n")
        
        # 按t_layer分组
        t_layer_groups = {}
        for name in sorted(transformer_groups.keys()):
            if name.startswith('T'):
                t_match = re.search(r'T(\d+)', name)
                if t_match:
                    t_layer = int(t_match.group(1))
                    if t_layer not in t_layer_groups:
                        t_layer_groups[t_layer] = []
                    t_layer_groups[t_layer].append(name)
        
        # 按t_layer显示
        for t_layer in sorted(t_layer_groups.keys()):
            t_names = t_layer_groups[t_layer]
            
            print(f"{'='*60}")
            print(f"Transformer Layer {t_layer}")
            print(f"{'='*60}")
            
            # 按step分组
            step_groups = {}
            for name in t_names:
                step_match = re.search(r'T\d+\.([^.]+)', name)
                if step_match:
                    step_name = step_match.group(1)
                    if step_name not in step_groups:
                        step_groups[step_name] = []
                    step_groups[step_name].append(name)
            
            # 按step显示
            for step_name in sorted(step_groups.keys()):
                step_names_list = step_groups[step_name]
                
                print(f"\n  {step_name}:")
                
                step_means = []
                for name in sorted(step_names_list):
                    gamma = transformer_groups[name]
                    mean_val = gamma.mean().item()
                    abs_mean = abs(mean_val)
                    growth = mean_val / init_val
                    abs_growth = abs(growth)
                    
                    all_means.append(mean_val)
                    all_abs_means.append(abs_mean)
                    all_growth.append(growth)
                    step_means.append(abs_mean)
                    
                    # 方向标记
                    direction = "➕" if mean_val >= 0 else "➖"
                    
                    # 释放程度标记（按绝对值）
                    if abs_growth < 2:
                        status = "🔒"
                    elif abs_growth < 10:
                        status = "🔓"
                    elif abs_growth < 100:
                        status = "📈"
                    else:
                        status = "🚀"
                    
                    # 简化显示：去掉T层前缀
                    short_name = re.sub(r'T\d+\.[^.]+\.', '', name)
                    print(f"    {direction} {short_name:30s}: {mean_val:9.6f} ({growth:7.1f}x, |{abs_growth:6.1f}x|) {status}")
                
                # Step统计
                if step_means:
                    step_avg = sum(step_means) / len(step_means)
                    print(f"    → 平均|gamma|: {step_avg:.6f} ({step_avg/init_val:.1f}x)")
            
            print()
    
    else:
        # 原有的显示方式（HT/2nn/2nns）
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
        print(f"\n✅ Transformer已充分释放（平均|gamma|={avg_abs_growth:.1f}x）")
    
    return avg_abs_growth, abs_growth_list

# 硬编码默认模型
default_model = "outputs/xps/htt50/checkpoint.th"

# 官方预训练HT模型路径
pretrained_model = r"C:\Users\35246\.cache\torch\hub\checkpoints\955717e8-8726e21a.th"

# 检查官方预训练模型作为参考标准
reference_avg = None
if Path(pretrained_model).exists():
    print("="*60)
    print("参考标准：官方预训练HT模型")  
    print("="*60)
    reference_avg, _ = check_layerscale(pretrained_model, is_reference=True)
    print("\n")
else:
    print(f"⚠️  未找到官方预训练模型: {pretrained_model}\n")
    print("\n")

# 检查用户模型
if Path(default_model).exists():
    print("="*60)
    print("用户训练模型")
    print("="*60)
    user_avg, _ = check_layerscale(default_model, is_reference=False)
    
    # 如果有参考标准，进行对比
    if reference_avg:
        print(f"\n{'='*60}")
        print(f"📊 与官方模型对比")
        print(f"{'='*60}")
        ratio = user_avg / reference_avg
        print(f"用户模型平均|gamma|: {user_avg:.1f}x")
        print(f"官方模型平均|gamma|: {reference_avg:.1f}x")
        print(f"释放比例: {ratio*100:.1f}%")
        
        if ratio < 0.3:
            print(f"⚠️  释放严重不足，建议继续训练")
        elif ratio < 0.6:
            print(f"🔓 释放不足，还有提升空间")
        elif ratio < 0.9:
            print(f"📈 释放良好，接近官方水平")
        else:
            print(f"✅ 释放充分，达到或超过官方水平")
else:
    print(f"❌ 默认模型不存在: {default_model}")
    print("\n用法: python check_layerscale.py [model_path]")
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        if Path(model_path).exists():
            check_layerscale(model_path, is_reference=False)
        else:
            print(f"❌ 文件不存在: {model_path}")

