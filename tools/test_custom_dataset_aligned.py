#!/usr/bin/env python3
# 严格按照test_pretrained.py流程对齐的自定义数据集评估脚本

from argparse import ArgumentParser
import logging
import sys
from pathlib import Path

import torch

from demucs import train, pretrained, evaluate
from demucs.audio import convert_audio
import torchaudio


def create_custom_solver(model_name, repo_path, test_data_path, device='cuda'):
    """创建与test_pretrained.py对齐的solver配置"""
    
    # 模拟test_pretrained.py的配置创建过程
    overrides = [
        f'dset.wav={test_data_path}',  # 设置测试数据路径
        'dset.samplerate=44100',
        'test.sdr=False',  # 使用新SDR (NSDR)
        'test.split=True',
        'test.overlap=0.25'
    ]
    
    # 获取XP配置（与test_pretrained.py相同）
    xp = train.main.get_xp(overrides)
    
    with xp.enter():
        # 创建solver，但只用于模型，不创建数据集（与test_pretrained.py相同）
        solver = train.get_solver(xp.cfg, model_only=True)
        
        # 加载预训练模型（与test_pretrained.py相同）
        # 创建完整的args对象，包含所有必需的属性
        class Args:
            def __init__(self, name, repo):
                self.name = name
                self.repo = repo
                self.sig = None  # 添加sig属性
        
        args = Args(model_name, repo_path)
        model = pretrained.get_model_from_args(args)
        solver.model = model.to(solver.device)
        solver.model.eval()
        
        return solver, xp.cfg


def evaluate_custom_dataset(model_name, repo_path, test_data_path, output_path=None):
    """使用与test_pretrained.py完全相同的评估流程"""
    
    torch.set_num_threads(1)
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    
    print(f"正在评估模型: {model_name}")
    print(f"模型仓库路径: {repo_path}")
    print(f"测试数据路径: {test_data_path}")
    print("="*60)
    
    try:
        # 创建solver（严格按照test_pretrained.py流程）
        solver, cfg = create_custom_solver(model_name, repo_path, test_data_path)
        
        print(f"使用设备: {solver.device}")
        print(f"模型源: {solver.model.sources}")
        print(f"采样率: {solver.model.samplerate}")
        print("="*60)
        
        # 执行评估（与test_pretrained.py完全相同）
        with torch.no_grad():
            results = evaluate.evaluate(solver, cfg.test.sdr)
        
        # 输出结果
        print("\n评估结果:")
        print("="*60)
        
        if isinstance(results, dict):
            for source, metrics in results.items():
                if isinstance(metrics, dict):
                    print(f"{source}:")
                    for metric_name, value in metrics.items():
                        if isinstance(value, (int, float)):
                            print(f"  {metric_name}: {value:.4f} dB")
                else:
                    print(f"{source}: {metrics:.4f} dB")
        else:
            print(f"Overall NSDR: {results:.4f} dB")
        
        # 保存结果
        if output_path:
            output_file = Path(output_path) / f"{model_name}_evaluation_aligned.txt"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"模型: {model_name}\n")
                f.write(f"模型仓库: {repo_path}\n")
                f.write(f"测试数据: {test_data_path}\n")
                f.write(f"评估方法: 严格按照test_pretrained.py流程\n")
                f.write(f"SDR类型: 新SDR (NSDR)\n")
                f.write("="*60 + "\n")
                f.write(f"结果: {results}\n")
            
            print(f"\n结果已保存到: {output_file}")
        
        return results
        
    except Exception as e:
        print(f"评估过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = ArgumentParser("test_custom_dataset_aligned",
                           description="按照test_pretrained.py流程评估自定义数据集")
    
    # 硬编码默认值
    parser.add_argument('--repo', default='./release_models',
                       help='模型仓库路径 (硬编码: ./release_models)')
    parser.add_argument('-n', '--name', default='3fba3fc3',
                       help='预训练模型名称 (硬编码: 3fba3fc3)')
    parser.add_argument('--data', default='data/instrumental_separation/test',
                       help='测试数据路径 (硬编码: data/instrumental_separation/test)')
    parser.add_argument('--output', default='./evaluation_results',
                       help='结果输出路径 (默认: ./evaluation_results)')
    
    args = parser.parse_args()
    
    # 验证路径
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"错误: 测试数据路径不存在: {data_path}")
        sys.exit(1)
    
    # 验证模型仓库路径
    if args.repo:
        repo_path = Path(args.repo)
        if not repo_path.exists():
            print(f"错误: 模型仓库路径不存在: {repo_path}")
            sys.exit(1)
    
    # 执行评估
    results = evaluate_custom_dataset(
        model_name=args.name,
        repo_path=args.repo,
        test_data_path=str(data_path),
        output_path=args.output
    )
    
    if results is not None:
        print("\n评估完成!")
    else:
        print("\n评估失败!")
        sys.exit(1)


if __name__ == '__main__':
    main()