#!/usr/bin/env python3
"""
通用脚本：检查多分辨率模型的融合权重
支持：
- htdemucs_n: 单组全局权重
- htdemucs_nn: 两组权重（频域+时域）
- htdemucs_ng: 全局权重
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import sys

def analyze_weight_group(raw_weights, name="融合权重", nfft_list=None):
    """分析一组权重"""
    normalized_weights = F.softmax(raw_weights, dim=0)
    num_resolutions = len(raw_weights)
    
    print(f"\n{'='*60}")
    print(f"📊 {name}")
    print(f"{'='*60}")
    print(f"分辨率数量: {num_resolutions}")
    print(f"原始权重: {raw_weights}")
    print(f"归一化权重: {normalized_weights}")
    
    # 生成分辨率标签
    if nfft_list is not None and len(nfft_list) == num_resolutions:
        resolutions = [f'{nfft}' for nfft in nfft_list]
    else:
        resolutions = [f'Res_{i+1}' for i in range(num_resolutions)]
    
    print(f"\n🎯 权重分布:")
    for i, (res, weight) in enumerate(zip(resolutions, normalized_weights)):
        bar = '█' * int(weight * 50)
        print(f"  {res:8s}: {weight:.4f} ({weight*100:.1f}%) {bar}")
    
    # 找出最偏好的分辨率
    max_idx = normalized_weights.argmax()
    print(f"\n🏆 最偏好: {resolutions[max_idx]} ({normalized_weights[max_idx]*100:.1f}%)")
    
    # 检查权重是否均匀分布
    uniform_weight = 1.0 / num_resolutions
    is_uniform = torch.allclose(normalized_weights, torch.ones(num_resolutions) * uniform_weight, atol=1e-3)
    if is_uniform:
        print("⚠️  权重接近均匀分布，可能还未充分训练")
    else:
        print("✅ 权重已分化，模型正在学习分辨率偏好")
    
    # 计算权重熵
    entropy = -(normalized_weights * torch.log(normalized_weights + 1e-8)).sum()
    max_entropy = torch.log(torch.tensor(float(num_resolutions)))
    entropy_ratio = entropy / max_entropy
    print(f"\n📈 权重熵: {entropy:.4f} / {max_entropy:.4f} ({entropy_ratio*100:.1f}%)")
    if entropy_ratio > 0.95:
        print("   → 分布很均匀")
    elif entropy_ratio > 0.8:
        print("   → 分布较均匀，有轻微偏好")
    else:
        print("   → 分布不均匀，有明显偏好")
    
    return normalized_weights

def check_fusion_weights(model_path):
    """检查模型的融合权重"""
    print(f"🔍 检查模型: {model_path}")
    
    try:
        # 加载模型检查点
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        if 'state' not in checkpoint:
            print("❌ 模型格式错误，未找到'state'字典")
            return
        
        state_dict = checkpoint['state']
        
        # 尝试获取NFFT列表
        nfft_list = None
        if 'args' in checkpoint:
            args = checkpoint['args']
            if hasattr(args, 'htdemucs_n') and hasattr(args.htdemucs_n, 'multi_freqs'):
                multi_freqs = args.htdemucs_n.multi_freqs
                if multi_freqs:
                    nfft_list = multi_freqs
            elif hasattr(args, 'htdemucs_nn') and hasattr(args.htdemucs_nn, 'multi_freqs'):
                multi_freqs = args.htdemucs_nn.multi_freqs
                if multi_freqs:
                    nfft_list = multi_freqs
        
        # 检查不同类型的融合权重
        found_weights = False
        
        # 1. 检查单组全局权重 (htdemucs_n)
        # 注意：如果同时有final_fusion_weights，说明是nn模型，会在后面处理
        if 'fusion_weights' in state_dict and 'final_fusion_weights' not in state_dict:
            found_weights = True
            print("\n✅ 发现全局融合权重 (htdemucs_n)")
            raw_weights = state_dict['fusion_weights']
            analyze_weight_group(raw_weights, "全局融合权重", nfft_list)
            
            # 检查EMA权重
            if 'weight_ema' in state_dict:
                ema_weights = state_dict['weight_ema']
                normalized_weights = F.softmax(raw_weights, dim=0)
                diff = (normalized_weights - ema_weights).abs().max()
                print(f"\n📈 EMA权重: {ema_weights}")
                print(f"原始权重与EMA差异: {diff:.6f}")
        
        # 2. 检查nn模型的双组权重
        if 'final_fusion_weights' in state_dict:
            found_weights = True
            print("\n✅ 发现htdemucs_nn模型（双权重结构）")
            
            # 瓶颈处的融合权重（全局）
            if 'fusion_weights' in state_dict:
                bottleneck_weights = state_dict['fusion_weights']
                analyze_weight_group(bottleneck_weights, "权重1: 瓶颈融合（频域）", nfft_list)
            
            # 最终输出的源特异性融合权重
            final_weights = state_dict['final_fusion_weights']
            final_norm = F.softmax(final_weights, dim=-1)
            
            print(f"\n{'='*60}")
            print(f"📊 权重2: 源特异性融合（时域）")
            print(f"{'='*60}")
            print(f"形状: {final_weights.shape[0]}个源 × {final_weights.shape[1]}个分辨率")
            
            # 生成分辨率标签
            if nfft_list is not None and len(nfft_list) == final_weights.shape[1]:
                resolutions = [f'{nfft}' for nfft in nfft_list]
            else:
                resolutions = [f'Res_{i+1}' for i in range(final_weights.shape[1])]
            
            # 简洁显示每个源的权重
            source_names = ['Drums', 'Bass', 'Other', 'Vocals']
            print(f"\n各源的权重分布:")
            for i, source in enumerate(source_names[:final_weights.shape[0]]):
                weights = final_norm[i]
                weight_str = " | ".join([f"{res}: {w:.1f}%" for res, w in zip(resolutions, weights * 100)])
                print(f"  {source:8s}: {weight_str}")
            
            # 计算平均权重
            final_avg = final_norm.mean(dim=0)
            avg_str = " | ".join([f"{res}: {w:.1f}%" for res, w in zip(resolutions, final_avg * 100)])
            print(f"  {'平均':8s}: {avg_str}")
            
            # 对比两组权重
            if 'fusion_weights' in state_dict:
                bottleneck_norm = F.softmax(bottleneck_weights, dim=0)
                diff = (bottleneck_norm - final_avg).abs()
                
                print(f"\n🔄 两组权重对比:")
                print(f"  瓶颈权重: {' | '.join([f'{w:.1f}%' for w in bottleneck_norm * 100])}")
                print(f"  最终平均: {' | '.join([f'{w:.1f}%' for w in final_avg * 100])}")
                print(f"  最大差异: {diff.max():.1f}%")
                
                if diff.max() < 0.05:
                    print("  ⚠️  两组权重接近，源特异性不明显")
                else:
                    print("  ✅ 两组权重有差异，不同源有不同偏好")
        
        if not found_weights:
            print("❌ 未找到融合权重，这可能不是多分辨率模型")
        
        if not found_weights:
            print("\n❌ 未找到融合权重")
            # 只在未找到权重时显示调试信息
            print("\n🔍 包含'fusion'的键:")
            fusion_keys = [key for key in sorted(state_dict.keys()) if 'fusion' in key.lower()]
            if fusion_keys:
                for key in fusion_keys:
                    print(f"  - {key}")
            else:
                print("  (未找到)")
                print("\n前20个state_dict键:")
                for key in sorted(state_dict.keys())[:20]:
                    print(f"  - {key}")
            
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()

def main():
    # 支持命令行参数
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        if Path(model_path).exists():
            check_fusion_weights(model_path)
        else:
            print(f"❌ 文件不存在: {model_path}")
        return
    
    # 默认模型
    default_model = "outputs/xps/248n97d170e1/best.th"
    
    if Path(default_model).exists():
        check_fusion_weights(default_model)
    else:
        print(f"❌ 默认模型不存在: {default_model}")
        print("\n用法: python check_model_weights.py <model_path>")

if __name__ == "__main__":
    main()