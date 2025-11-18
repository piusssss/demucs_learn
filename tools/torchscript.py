"""
导出 Demucs 模型为 TorchScript 格式
"""
import torch
from pathlib import Path
from demucs.pretrained import get_model


def export_torchscript(model_name='htdemucs_ft', output_dir='./torchscript_models'):
    """导出模型为 TorchScript（不优化，更稳定）"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"加载模型: {model_name}")
    bag = get_model(model_name)
    
    # 检查是否有 GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"导出设备: {device}")
    
    for i, model in enumerate(bag.models):
        model.eval()
        model.to(device)  # 在目标设备上导出
        
        segment_samples = int(float(model.segment) * model.samplerate)
        dummy_input = torch.randn(1, model.audio_channels, segment_samples).to(device)
        
        print(f"导出模型 {i+1}/{len(bag.models)}...")
        with torch.no_grad():
            traced = torch.jit.trace(model, dummy_input)
        
        output_path = output_dir / f"{model_name}_model{i}.pt"
        traced.save(str(output_path))
        print(f"  保存: {output_path}")
    
    print("完成")


if __name__ == '__main__':
    export_torchscript('htdemucs_ft', './torchscript_models')
