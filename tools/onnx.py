import torch
import os
from demucs.pretrained import get_model


def export_bag_to_onnx(bag_name='htdemucs_ft', output_dir='./onnx_models'):
    """
    导出 bag of models 到独立的 ONNX 文件
    
    Args:
        bag_name: 模型名称，如 'htdemucs', 'htdemucs_ft', 'mdx' 等
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载 bag
    print(f"加载模型包: {bag_name}")
    bag = get_model(bag_name)
    
    # 检查模型数量
    num_models = len(bag.models)
    print(f"模型包含 {num_models} 个子模型")
    
    # 准备 dummy input（10秒音频，44100采样率）
    dummy_input = torch.randn(1, 2, 44100*10)
    print(f"Dummy input shape: {dummy_input.shape}")
    
    # 导出每个模型
    for i, model in enumerate(bag.models):
        print(f"\n导出模型 {i+1}/{num_models}...")
        
        # 设置为评估模式
        model.eval()
        
        # 禁用 train_segment 模式以避免动态形状问题
        if hasattr(model, 'use_train_segment'):
            model.use_train_segment = False
        
        # 输出文件名
        output_path = os.path.join(output_dir, f'{bag_name}_model{i}.onnx')
        
        try:
            # 导出 ONNX
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                input_names=['mix'],
                output_names=['output'],
                dynamic_axes={
                    'mix': {2: 'length'},
                    'output': {3: 'length'}
                },
                opset_version=17,
                do_constant_folding=True,
                verbose=False
            )
            print(f"✅ 成功导出: {output_path}")
            
        except Exception as e:
            print(f"❌ 导出失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # 保存元信息
    import json
    meta = {
        'bag_name': bag_name,
        'num_models': num_models,
        'sources': bag.sources,
        'samplerate': bag.samplerate,
        'audio_channels': bag.audio_channels,
        'weights': bag.weights,
    }
    meta_path = os.path.join(output_dir, f'{bag_name}_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"\n✅ 元信息保存到: {meta_path}")
    
    return True


if __name__ == '__main__':
    # 导出模型
    bag_name = 'htdemucs_ft'  # 可以改成 'htdemucs', 'mdx' 等
    output_dir = './onnx_models'
    
    
    success = export_bag_to_onnx(bag_name, output_dir)
    
    if success:
        print("\n" + "="*50)
        print("✅ 导出完成")
        print("="*50)
    else:
        print("\n" + "="*50)
        print("❌ 导出失败")

