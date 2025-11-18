"""
测试 TorchScript 模型
复用原版预处理逻辑，只替换推理部分
"""
import torch
import torchaudio
from pathlib import Path
from demucs.apply import apply_model, BagOfModels
import torch.nn as nn
from demucs.audio import save_audio


class TorchScriptBag(BagOfModels):
    """用 TorchScript 模型替换原版模型的 Bag"""
    def __init__(self, model_dir, model_name='htdemucs_ft'):
        model_dir = Path(model_dir)
        
        # 加载所有子模型
        models = []
        i = 0
        while True:
            model_path = model_dir / f"{model_name}_model{i}.pt"
            if not model_path.exists():
                break
            print(f"加载: {model_path}")
            model = torch.jit.load(str(model_path))
            model.eval()
            models.append(model)
            i += 1
        
        print(f"加载了 {len(models)} 个模型")
        
        # htdemucs_ft 的权重配置
        weights = [
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
        ]
        
        # 调用父类初始化（会自动设置 models, sources, samplerate 等）
        # 但 TorchScript 模型没有这些属性，所以先手动设置
        for model in models:
            model.sources = ['drums', 'bass', 'other', 'vocals']
            model.samplerate = 44100
            model.audio_channels = 2
            model.segment = 7.8
            # 添加 valid_length 方法
            model.valid_length = lambda length: int(7.8 * 44100)
        
        super().__init__(models, weights)
        

def test_torchscript(audio_path, output_dir='./separated_torchscript', two_stems=None):
    """
    使用 TorchScript 模型进行音频分离测试
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("TorchScript 模型测试")
    print("="*60)
    
    # 1. 加载 TorchScript 模型
    print("\n[1/4] 加载 TorchScript 模型...")
    bag = TorchScriptBag('./torchscript_models', 'htdemucs_ft')
    
    # 2. 加载音频
    print(f"\n[2/4] 加载音频: {audio_path}")
    wav, sr = torchaudio.load(audio_path)
    
    # 重采样到 44100
    if sr != bag.samplerate:
        print(f"  重采样: {sr} -> {bag.samplerate}")
        wav = torchaudio.functional.resample(wav, sr, bag.samplerate)
    
    # 转为双声道
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    elif wav.shape[0] > 2:
        wav = wav[:2]
    
    # 添加 batch 维度
    wav = wav.unsqueeze(0)
    print(f"  音频形状: {wav.shape}")
    print(f"  时长: {wav.shape[-1] / bag.samplerate:.1f} 秒")
    
    # 3. 使用 apply_model 进行分离（复用原版逻辑）
    print(f"\n[3/4] 分离音频...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  设备: {device}")
    
    # apply_model 会自动管理每个子模型的设备迁移（一次只加载一个到 GPU）
    with torch.no_grad():
        separated = apply_model(
            bag,
            wav,
            shifts=1,
            split=True,
            overlap=0.25,
            device=device
        )
    
    print(f"  输出形状: {separated.shape}")
    
    # 4. 保存结果
    print(f"\n[4/4] 保存结果到: {output_dir}")
    separated = separated.cpu()
    
    if two_stems is not None:
        # two-stems 模式：只输出指定音源和其他音源的混合
        print(f"  Two-stems 模式: {two_stems}")
        
        # 找到目标音源的索引
        if two_stems not in bag.sources:
            raise ValueError(f"音源 '{two_stems}' 不在 {bag.sources} 中")
        
        target_idx = bag.sources.index(two_stems)
        
        # 保存目标音源
        output_path = output_dir / f"{two_stems}.wav"
        save_audio(separated[0, target_idx], str(output_path), bag.samplerate)
        print(f"  保存: {output_path}")
        
        # 保存其他音源的混合（no_{target}）- 使用官方默认的 add 方法
        other_sources = torch.zeros_like(separated[0, 0])
        for i, source in enumerate(bag.sources):
            if i != target_idx:
                other_sources += separated[0, i]
        
        output_path = output_dir / f"no_{two_stems}.wav"
        save_audio(other_sources, str(output_path), bag.samplerate)
        print(f"  保存: {output_path}")
    else:
        # 正常模式：保存所有音源
        for i, source in enumerate(bag.sources):
            output_path = output_dir / f"{source}.wav"
            save_audio(
                separated[0, i],
                str(output_path),
                bag.samplerate
            )
            print(f"  保存: {output_path}")
    
    print("\n" + "="*60)
    print("✅ 测试完成")
    print("="*60)


if __name__ == '__main__':
    # 硬编码配置
    audio_path = 'music/back.wav'
    output_dir = './separated_torchscript'
    two_stems = 'vocals'  # 可选: None, 'drums', 'bass', 'other', 'vocals'
    
    # 检查文件是否存在
    if not Path(audio_path).exists():
        print(f"错误: 找不到测试音频 {audio_path}")
        print("请修改 audio_path 为实际的音频文件路径")
    else:
        test_torchscript(audio_path, output_dir, two_stems)
