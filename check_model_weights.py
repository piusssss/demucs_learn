#!/usr/bin/env python3
"""
简单脚本：检查HTDemucs_n模型的融合权重
"""

import torch
import torch.nn.functional as F
from pathlib import Path

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
        
        # 检查融合权重
        if 'fusion_weights' not in state_dict:
            print("❌ 未找到融合权重，这可能不是多分辨率模型")
            return
        
        # 获取权重
        raw_weights = state_dict['fusion_weights']
        normalized_weights = F.softmax(raw_weights, dim=0)
        
        # 动态获取分辨率数量
        num_resolutions = len(raw_weights)
        
        print("✅ 模型加载成功!")
        print(f"\n📊 融合权重分析:")
        print(f"分辨率数量: {num_resolutions}")
        print(f"原始权重: {raw_weights}")
        print(f"归一化权重: {normalized_weights}")
        
        # 尝试从模型配置中获取实际的NFFT值
        nfft_list = None
        if 'nfft_list' in checkpoint:
            nfft_list = checkpoint['nfft_list']
        elif hasattr(checkpoint.get('model', None), 'nfft_list'):
            nfft_list = checkpoint['model'].nfft_list
        
        # 生成分辨率标签
        if nfft_list is not None:
            resolutions = [f'{nfft}Hz' for nfft in nfft_list]
        else:
            # 默认标签
            resolutions = [f'Resolution_{i+1}' for i in range(num_resolutions)]
        
        print(f"\n🎯 分辨率权重分布:")
        for i, (res, weight) in enumerate(zip(resolutions, normalized_weights)):
            print(f"  {res}: {weight:.4f} ({weight*100:.1f}%)")
        
        # 找出最偏好的分辨率
        max_idx = normalized_weights.argmax()
        print(f"\n🏆 最偏好分辨率: {resolutions[max_idx]} ({normalized_weights[max_idx]*100:.1f}%)")
        
        # 检查权重是否均匀分布
        uniform_weight = 1.0 / num_resolutions
        is_uniform = torch.allclose(normalized_weights, torch.ones(num_resolutions) * uniform_weight, atol=1e-3)
        if is_uniform:
            print("⚠️  权重接近均匀分布，可能还未充分训练")
        else:
            print("✅ 权重已开始分化，模型正在学习分辨率偏好")
        
        # 检查EMA权重（如果存在）
        if 'weight_ema' in state_dict:
            ema_weights = state_dict['weight_ema']
            print(f"\n📈 EMA权重: {ema_weights}")
            
            # 比较原始权重和EMA权重的差异
            diff = (normalized_weights - ema_weights).abs().max()
            print(f"原始权重与EMA权重最大差异: {diff:.6f}")
        
        # 计算权重熵（衡量分布均匀程度）
        entropy = -(normalized_weights * torch.log(normalized_weights + 1e-8)).sum()
        max_entropy = torch.log(torch.tensor(float(num_resolutions)))  # 均匀分布的最大熵
        print(f"\n📈 权重熵: {entropy:.4f} / {max_entropy:.4f} ({entropy/max_entropy*100:.1f}%)")
        if entropy/max_entropy > 0.95:
            print("   → 权重分布很均匀")
        elif entropy/max_entropy > 0.8:
            print("   → 权重分布较均匀，有轻微偏好")
        else:
            print("   → 权重分布不均匀，有明显偏好")
            
    except Exception as e:
        print(f"❌ 错误: {e}")

def main():
    # 默认检查路径
    default_path = "outputs/xps/48f6aa15"
    
    # 查找模型文件
    base_dir = Path(default_path)
    possible_files = ["checkpoint.th", "best.th"]
    
    model_file = None
    for filename in possible_files:
        filepath = base_dir / filename
        if filepath.exists():
            model_file = filepath
            break
    
    if model_file:
        check_fusion_weights(model_file)
    else:
        print(f"❌ 在 {base_dir} 中未找到模型文件")
        if base_dir.exists():
            print("目录内容:")
            for item in base_dir.iterdir():
                print(f"  - {item.name}")

if __name__ == "__main__":
    main()