#!/usr/bin/env python3
"""
重新计算instrumental评估结果的汇总统计
处理NaN和Infinity值，生成正确的统计数据
"""

import json
import numpy as np
from pathlib import Path
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clean_values(values):
    """
    清理数值列表，移除NaN和Infinity值
    
    Args:
        values: 数值列表
        
    Returns:
        cleaned_values: 清理后的有效数值列表
    """
    cleaned = []
    for val in values:
        if isinstance(val, (int, float)) and np.isfinite(val) and not np.isnan(val):
            cleaned.append(val)
    return cleaned

def calculate_statistics(values):
    """
    计算统计数据
    
    Args:
        values: 数值列表
        
    Returns:
        stats: 包含均值、中位数、标准差的字典
    """
    if not values:
        return {
            'mean': None,
            'median': None,
            'std': None,
            'count': 0
        }
    
    values = np.array(values)
    return {
        'mean': float(np.mean(values)),
        'median': float(np.median(values)),
        'std': float(np.std(values)),
        'count': len(values)
    }

def calculate_median_of_medians(track_results, metric):
    """
    计算每个轨道中位数的中位数
    
    Args:
        track_results: 轨道结果字典
        metric: 指标名称 (SDR, ISR, SIR, SAR)
        
    Returns:
        median_of_medians: 中位数的中位数
    """
    track_medians = []
    
    for track_name, scores in track_results.items():
        if metric in scores:
            cleaned_values = clean_values(scores[metric])
            if cleaned_values:
                track_median = np.median(cleaned_values)
                track_medians.append(track_median)
    
    if track_medians:
        return float(np.median(track_medians))
    else:
        return None

def recalculate_summary(input_file, output_file=None):
    """
    重新计算汇总统计
    
    Args:
        input_file: 输入的评估结果JSON文件
        output_file: 输出文件路径，如果为None则覆盖原文件
    """
    logger.info(f"Loading evaluation results from {input_file}")
    
    # 加载原始结果
    with open(input_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    if 'tracks' not in results:
        logger.error("No 'tracks' section found in results file")
        return
    
    track_results = results['tracks']
    logger.info(f"Found {len(track_results)} tracks")
    
    # 收集所有指标的数值
    all_metrics = {
        'sdr': [],
        'isr': [],
        'sir': [],
        'sar': []
    }
    
    # 统计每个指标的有效数据
    metric_stats = {}
    
    for track_name, scores in track_results.items():
        for metric_key, metric_name in [('SDR', 'sdr'), ('ISR', 'isr'), ('SIR', 'sir'), ('SAR', 'sar')]:
            if metric_key in scores:
                cleaned_values = clean_values(scores[metric_key])
                all_metrics[metric_name].extend(cleaned_values)
                
                logger.info(f"{track_name} - {metric_key}: {len(cleaned_values)} valid values out of {len(scores[metric_key])}")
    
    # 计算汇总统计
    summary = {}
    
    for metric_name in ['sdr', 'isr', 'sir', 'sar']:
        logger.info(f"Calculating statistics for {metric_name.upper()}")
        logger.info(f"Total valid {metric_name.upper()} values: {len(all_metrics[metric_name])}")
        
        # 计算基本统计
        stats = calculate_statistics(all_metrics[metric_name])
        
        # 计算中位数的中位数
        median_of_medians = calculate_median_of_medians(track_results, metric_name.upper())
        
        # 添加到汇总
        summary[f'{metric_name}_mean'] = stats['mean']
        summary[f'{metric_name}_median'] = stats['median']
        summary[f'{metric_name}_std'] = stats['std']
        summary[f'{metric_name}_median_of_medians'] = median_of_medians
        summary[f'{metric_name}_count'] = stats['count']
        
        # 打印统计信息
        if stats['mean'] is not None:
            logger.info(f"{metric_name.upper()} - Mean: {stats['mean']:.3f}, Median: {stats['median']:.3f}, Std: {stats['std']:.3f}")
            if median_of_medians is not None:
                logger.info(f"{metric_name.upper()} - Median of Medians: {median_of_medians:.3f}")
            else:
                logger.info(f"{metric_name.upper()} - Median of Medians: N/A")
        else:
            logger.warning(f"{metric_name.upper()} - No valid data found")
    
    # 更新结果
    results['summary'] = summary
    
    # 保存结果
    if output_file is None:
        output_file = input_file
    
    logger.info(f"Saving updated results to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info("Summary recalculation completed successfully!")
    
    # 打印最终汇总
    print("\n" + "="*60)
    print("INSTRUMENTAL SEPARATION EVALUATION SUMMARY")
    print("="*60)
    
    for metric_name in ['sdr', 'isr', 'sir', 'sar']:
        metric_upper = metric_name.upper()
        mean_val = summary.get(f'{metric_name}_mean')
        median_val = summary.get(f'{metric_name}_median')
        std_val = summary.get(f'{metric_name}_std')
        count_val = summary.get(f'{metric_name}_count', 0)
        
        print(f"\n{metric_upper}:")
        if mean_val is not None and median_val is not None and std_val is not None:
            print(f"  Mean:   {mean_val:.3f} dB")
            print(f"  Median: {median_val:.3f} dB")
            print(f"  Std:    {std_val:.3f} dB")
            print(f"  Count:  {count_val} valid values")
        else:
            print(f"  No valid data available")
    
    print("\n" + "="*60)

def main():
    """主函数"""
    input_file = "instrumental_evaluation_results.json"
    output_file = "instrumental_evaluation_results_fixed.json"
    
    if not Path(input_file).exists():
        logger.error(f"Input file {input_file} not found")
        return
    
    try:
        recalculate_summary(input_file, output_file)
    except Exception as e:
        logger.error(f"Error during recalculation: {e}")
        raise

if __name__ == "__main__":
    main()