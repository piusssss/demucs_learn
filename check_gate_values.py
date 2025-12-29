#!/usr/bin/env python3
"""
检查 NanoFusionHead 的 Gate Sigmoid 平均值
用于验证 gate 是否学习到有效的置信度，还是退化到接近 0
"""

import torch
from pathlib import Path
import sys

def check_gate_values(model_path):
    """检查模型的 gate 参数值"""
    print(f"🔍 检查模型: {model_path}")
    
    try:
        # 加载模型检查点
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        if 'state' not in checkpoint:
            print("❌ 模型格式错误，未找到'state'字典")
            return
        
        state_dict = checkpoint['state']
        
        # 查找 gate 相关的参数
        gate_keys = [k for k in state_dict.keys() if 'fusion.gate' in k]
        
        if not gate_keys:
            print("❌ 未找到 NanoFusionHead 的 gate 参数")
            print("   这个模型可能不是 HTDemucs_2nns 或未使用 NanoFusionHead")
            return
        
        print(f"\n{'='*60}")
        print(f"📊 NanoFusionHead Gate 参数分析")
        print(f"{'='*60}")
        print(f"找到 {len(gate_keys)} 个 gate 参数\n")
        
        # 分析每个 gate 参数
        gate_stats = {}
        
        for key in sorted(gate_keys):
            param = state_dict[key]
            
            # 计算统计信息
            mean_val = param.mean().item()
            std_val = param.std().item()
            min_val = param.min().item()
            max_val = param.max().item()
            
            gate_stats[key] = {
                'mean': mean_val,
                'std': std_val,
                'min': min_val,
                'max': max_val,
                'shape': tuple(param.shape)
            }
            
            print(f"🔧 {key}")
            print(f"   形状: {tuple(param.shape)}")
            print(f"   均值: {mean_val:+.6f}")
            print(f"   标准差: {std_val:.6f}")
            print(f"   范围: [{min_val:+.6f}, {max_val:+.6f}]")
            print()
        
        # 模拟 sigmoid 后的输出
        print(f"{'='*60}")
        print(f"🎯 模拟 Sigmoid 输出 (置信度)")
        print(f"{'='*60}\n")
        
        for key in sorted(gate_keys):
            if 'weight' in key:  # 只分析权重，不分析 bias
                param = state_dict[key]
                
                # 获取对应的 bias
                bias_key = key.replace('weight', 'bias')
                bias = state_dict.get(bias_key, torch.zeros(param.shape[0]))
                
                # 模拟一个随机输入通过 gate
                # 假设输入是标准正态分布
                dummy_input = torch.randn(1, param.shape[1], 1000)  # [B, C, T]
                gate_output = torch.conv1d(dummy_input, param, bias, 
                                          groups=param.shape[0] // bias.shape[0] if bias.shape[0] < param.shape[0] else 1,
                                          padding=1)
                gate_sigmoid = torch.sigmoid(gate_output)
                
                sigmoid_mean = gate_sigmoid.mean().item()
                sigmoid_std = gate_sigmoid.std().item()
                sigmoid_min = gate_sigmoid.min().item()
                sigmoid_max = gate_sigmoid.max().item()
                
                print(f"🎲 {key.replace('fusion.gate.', '')}")
                print(f"   Sigmoid 均值: {sigmoid_mean:.4f} ({sigmoid_mean*100:.1f}%)")
                print(f"   Sigmoid 标准差: {sigmoid_std:.4f}")
                print(f"   Sigmoid 范围: [{sigmoid_min:.4f}, {sigmoid_max:.4f}]")
                
                # 评估置信度
                if sigmoid_mean < 0.1:
                    status = "❌ 极低 - Gate 几乎不起作用"
                elif sigmoid_mean < 0.3:
                    status = "⚠️  偏低 - Gate 较保守"
                elif sigmoid_mean < 0.7:
                    status = "✅ 正常 - Gate 在合理范围"
                elif sigmoid_mean < 0.9:
                    status = "⚠️  偏高 - Gate 较激进"
                else:
                    status = "❌ 极高 - Gate 几乎总是采纳修正"
                
                print(f"   状态: {status}")
                
                # 生成可视化条形图
                bar_length = int(sigmoid_mean * 50)
                bar = '█' * bar_length + '░' * (50 - bar_length)
                print(f"   [{bar}] {sigmoid_mean:.2%}")
                print()
        
        # 总结
        print(f"{'='*60}")
        print(f"📝 总结")
        print(f"{'='*60}")
        
        # 计算所有 weight 参数的平均 sigmoid 输出
        all_sigmoid_means = []
        for key in sorted(gate_keys):
            if 'weight' in key:
                param = state_dict[key]
                bias_key = key.replace('weight', 'bias')
                bias = state_dict.get(bias_key, torch.zeros(param.shape[0]))
                
                dummy_input = torch.randn(1, param.shape[1], 1000)
                gate_output = torch.conv1d(dummy_input, param, bias,
                                          groups=param.shape[0] // bias.shape[0] if bias.shape[0] < param.shape[0] else 1,
                                          padding=1)
                gate_sigmoid = torch.sigmoid(gate_output)
                all_sigmoid_means.append(gate_sigmoid.mean().item())
        
        if all_sigmoid_means:
            overall_mean = sum(all_sigmoid_means) / len(all_sigmoid_means)
            print(f"\n🎯 整体 Gate 置信度: {overall_mean:.4f} ({overall_mean*100:.1f}%)")
            
            if overall_mean < 0.2:
                print("⚠️  Gate 非常保守，可能需要：")
                print("   1. 检查训练是否充分")
                print("   2. 考虑调整 gate 的初始化")
                print("   3. 验证 correction 分支是否有效")
            elif overall_mean > 0.8:
                print("⚠️  Gate 非常激进，可能需要：")
                print("   1. 检查是否过拟合")
                print("   2. 验证 correction 是否引入噪声")
            else:
                print("✅ Gate 工作正常，在合理的置信度范围内")
        
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        # 默认检查路径
        default_model = "release_models/97d170e1.th"
        if Path(default_model).exists():
            model_path = default_model
        else:
            print(f"❌ 默认模型不存在: {default_model}")
            print("\n用法: python check_gate_values.py <model_path>")
            print("示例: python check_gate_values.py release_models/your_model.th")
            return
    
    if not Path(model_path).exists():
        print(f"❌ 模型文件不存在: {model_path}")
        return
    
    check_gate_values(model_path)

if __name__ == "__main__":
    main()
