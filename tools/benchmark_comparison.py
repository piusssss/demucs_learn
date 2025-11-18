"""
对比 TorchScript 和原版命令行的速度差异
"""
import time
import subprocess
from pathlib import Path
from test_torchscript import test_torchscript


def main():
    # 配置
    audio_path = 'music/back.wav'
    two_stems = 'vocals'
    
    if not Path(audio_path).exists():
        print(f"错误: 找不到 {audio_path}")
        return
    
    print(f"测试文件: {audio_path}")
    print(f"分离音源: {two_stems}\n")
    
    # 测试 TorchScript
    print("=" * 50)
    print("测试 TorchScript 版本")
    print("=" * 50)
    start = time.time()
    test_torchscript(audio_path, './out_ts', two_stems)
    ts_time = time.time() - start
    print(f"\nTorchScript 耗时: {ts_time:.2f} 秒")
    
    # 等待
    time.sleep(3)
    
    # 测试原版
    print("\n" + "=" * 50)
    print("测试原版命令行")
    print("=" * 50)
    cmd = ['python', '-m', 'demucs.separate', '-n', 'htdemucs_ft', 
           '--two-stems', two_stems, '-o', './out_orig', audio_path]
    start = time.time()
    subprocess.run(cmd, capture_output=True)
    orig_time = time.time() - start
    print(f"\n原版耗时: {orig_time:.2f} 秒")
    
    # 对比
    print("\n" + "=" * 50)
    print("对比结果")
    print("=" * 50)
    print(f"TorchScript: {ts_time:.2f} 秒")
    print(f"原版:        {orig_time:.2f} 秒")
    
    if ts_time < orig_time:
        speedup = orig_time / ts_time
        print(f"\n✅ TorchScript 快 {(speedup-1)*100:.1f}% (加速 {speedup:.2f}x)")
    else:
        slowdown = ts_time / orig_time
        print(f"\n❌ TorchScript 慢 {(slowdown-1)*100:.1f}%")


if __name__ == '__main__':
    main()
