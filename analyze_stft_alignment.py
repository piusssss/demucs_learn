#!/usr/bin/env python3
"""
分析多分辨率 STFT 的时间对齐关系
可视化每个时间帧对应的真实时间范围
"""

import torch
import math
from demucs.htdemucs_mr import HTDemucs_mr

def analyze_stft_time_alignment():
    """分析 STFT 时间帧的对应关系"""
    
    # 创建一个简单的测试音频
    samplerate = 44100
    duration = 0.5  # 1 秒
    length = int(samplerate * duration)
    
    print(f"{'='*80}")
    print(f"多分辨率 STFT 时间对齐分析")
    print(f"{'='*80}")
    print(f"音频长度: {length} samples ({duration}s @ {samplerate}Hz)")
    print()
    
    # 创建测试音频 [B, C, T]
    mix = torch.randn(1, 2, length)
    
    # 创建模型实例（只用来调用 _spec 方法）
    model = HTDemucs_mr(
        sources=['vocals', 'drums', 'bass', 'other'],
        nfft_list=[2048, 4096, 8192],
        samplerate=samplerate
    )
    
    nfft_list = model.nfft_list
    hop_lengths = model.hop_lengths
    
    print(f"分辨率配置:")
    for i, (nfft, hop) in enumerate(zip(nfft_list, hop_lengths)):
        print(f"  Resolution {i}: nfft={nfft}, hop={hop}")
    print()
    
    # 对每个分辨率进行 STFT
    results = []
    
    for res_idx, (nfft, hop_length) in enumerate(zip(nfft_list, hop_lengths)):
        print(f"{'='*80}")
        print(f"Resolution {res_idx}: NFFT={nfft}, HOP={hop_length}")
        print(f"{'='*80}")
        
        # 调用 _spec 方法
        z = model._spec(mix, nfft=nfft, hop_length=hop_length)
        
        B, C, Fq, T = z.shape
        print(f"输出形状: [B={B}, C={C}, Freq={Fq}, Time={T}]")
        print()
        
        # 计算 padding 信息
        hl = hop_length
        le = int(math.ceil(length / hl))
        pad_left = hl
        pad_right = pad_left + le * hl - length
        padded_length = pad_left + length + pad_right
        
        print(f"Padding 信息:")
        print(f"  原始长度: {length}")
        print(f"  左侧 padding: {pad_left}")
        print(f"  右侧 padding: {pad_right}")
        print(f"  Padded 长度: {padded_length}")
        print(f"  期望帧数 le: {le}")
        print()
        
        # 计算每一帧的时间范围
        print(f"时间帧分析 (在 padded 坐标系中):")
        print(f"{'帧号':<8} {'窗口范围':<25} {'中心位置':<12} {'状态':<10} {'覆盖原始音频'}")
        print(f"{'-'*80}")
        
        frame_info = []
        
        # 计算所有帧（包括被丢弃的）
        total_frames = int(math.ceil(padded_length / hl))
        for frame_idx in range(total_frames):
            # 窗口在 padded 坐标系中的位置
            start_padded = frame_idx * hl
            end_padded = start_padded + nfft
            center_padded = start_padded + nfft // 2
            
            # 转换到原始坐标系（去掉 padding）
            start_original = start_padded - pad_left
            end_original = end_padded - pad_left
            center_original = center_padded - pad_left
            
            # 判断状态
            if frame_idx == 0:
                status = "丢弃"
            elif frame_idx > le:
                status = "丢弃"
            else:
                status = "保留"
            
            # 计算与原始音频的重叠
            overlap_start = max(0, start_original)
            overlap_end = min(length, end_original)
            
            if overlap_start < overlap_end:
                overlap_str = f"[{overlap_start}, {overlap_end})"
                overlap_len = overlap_end - overlap_start
                overlap_pct = overlap_len / nfft * 100
                overlap_str += f" ({overlap_pct:.1f}%)"
            else:
                overlap_str = "无重叠"
            
            print(f"Frame {frame_idx:<2} "
                  f"[{start_original:>6}, {end_original:>6})  "
                  f"中心:{center_original:>6}  "
                  f"{status:<10} "
                  f"{overlap_str}")
            
            if status == "保留":
                frame_info.append({
                    'frame_idx': frame_idx,
                    'start': start_original,
                    'end': end_original,
                    'center': center_original,
                    'overlap_start': overlap_start,
                    'overlap_end': overlap_end
                })
        
        print()
        print(f"保留的帧数: {len(frame_info)} (代码输出: {T})")
        assert len(frame_info) == T, f"帧数不匹配！计算={len(frame_info)}, 实际={T}"
        print()
        
        results.append({
            'nfft': nfft,
            'hop': hop_length,
            'frames': frame_info,
            'shape': (Fq, T)
        })
    
    # 分析跨分辨率的对应关系
    print(f"{'='*80}")
    print(f"跨分辨率时间对应关系分析")
    print(f"{'='*80}")
    print()
    
    # 以最高时间分辨率（最小 hop）为基准
    base_res = results[0]  # 2048-STFT
    
    for res_idx in range(1, len(results)):
        target_res = results[res_idx]
        
        print(f"对应关系: {base_res['nfft']}-STFT → {target_res['nfft']}-STFT")
        print(f"{'-'*80}")
        
        # 为每个 target 帧找到对应的 base 帧
        for target_frame in target_res['frames']:
            target_idx = target_frame['frame_idx']
            target_start = target_frame['overlap_start']
            target_end = target_frame['overlap_end']
            target_center = target_frame['center']
            
            # 找到所有与 target 帧重叠的 base 帧
            overlapping_base = []
            for base_frame in base_res['frames']:
                base_start = base_frame['overlap_start']
                base_end = base_frame['overlap_end']
                
                # 计算重叠区域
                overlap_start = max(target_start, base_start)
                overlap_end = min(target_end, base_end)
                
                if overlap_start < overlap_end:
                    overlap_len = overlap_end - overlap_start
                    # 计算重叠比例（相对于 target 帧）
                    overlap_ratio = overlap_len / (target_end - target_start)
                    overlapping_base.append({
                        'idx': base_frame['frame_idx'],
                        'center': base_frame['center'],
                        'overlap_ratio': overlap_ratio
                    })
            
            # 打印对应关系
            base_indices = [f"F{b['idx']}" for b in overlapping_base]
            base_centers = [f"{b['center']}" for b in overlapping_base]
            overlap_ratios = [f"{b['overlap_ratio']*100:.1f}%" for b in overlapping_base]
            
            print(f"{target_res['nfft']}-Frame{target_idx} (中心:{target_center:>6}) → "
                  f"{base_res['nfft']}-{base_indices} "
                  f"(中心:{base_centers}) "
                  f"重叠比例:{overlap_ratios}")
        
        print()
    
    # 分析 MR 的 repeat 策略
    print(f"{'='*80}")
    print(f"MR 的 repeat 策略分析")
    print(f"{'='*80}")
    print()
    
    for res_idx in range(1, len(results)):
        base_T = results[0]['shape'][1]
        target_T = results[res_idx]['shape'][1]
        
        if base_T % target_T == 0:
            time_repeat = base_T // target_T
            print(f"{results[res_idx]['nfft']}-STFT: {target_T} 帧 → repeat {time_repeat} 次 → {base_T} 帧")
            
            # 模拟 repeat
            repeat_mapping = []
            for i in range(target_T):
                repeat_mapping.extend([i] * time_repeat)
            
            print(f"  Repeat 映射: {repeat_mapping}")
            
            # 对比实际的对应关系
            print(f"  实际对应关系:")
            for base_idx in range(min(base_T, 10)):  # 只显示前10个
                target_idx_repeat = repeat_mapping[base_idx]
                
                # 找到实际最接近的 target 帧
                base_center = results[0]['frames'][base_idx]['center']
                min_dist = float('inf')
                target_idx_actual = 0
                for t_idx, t_frame in enumerate(results[res_idx]['frames']):
                    dist = abs(t_frame['center'] - base_center)
                    if dist < min_dist:
                        min_dist = dist
                        target_idx_actual = t_idx
                
                match = "✓" if target_idx_repeat == target_idx_actual else "✗"
                print(f"    {base_res['nfft']}-F{base_idx} (中心:{base_center:>6}): "
                      f"repeat→F{target_idx_repeat}, 实际→F{target_idx_actual} {match}")
        else:
            print(f"{results[res_idx]['nfft']}-STFT: {target_T} 帧不能被 {base_T} 整除")
            print(f"  需要先 pad 到 {math.ceil(target_T / base_T) * base_T} 帧")
        
        print()

if __name__ == "__main__":
    analyze_stft_time_alignment()
