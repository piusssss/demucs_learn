#!/usr/bin/env python3
"""
4-source music separation evaluation script using htdemucs model.
Evaluates drums, bass, other, and vocals separately on MUSDB18-HQ test set.
"""

import os
import logging
import numpy as np
import torch as th
import museval
from pathlib import Path
import soundfile as sf
from concurrent import futures
from dora.log import LogProgress
import argparse

# Import demucs modules
from demucs.pretrained import get_model
from demucs.apply import apply_model
from demucs.audio import convert_audio
from demucs.utils import DummyPoolExecutor

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_audio(path, samplerate=44100):
    """Load audio file and convert to tensor."""
    audio, sr = sf.read(path)
    if audio.ndim == 1:
        audio = audio[None, :]  # Add channel dimension
    else:
        audio = audio.T  # Convert to (channels, samples)
    
    audio = th.from_numpy(audio).float()
    
    # Convert sample rate if needed
    if sr != samplerate:
        audio = convert_audio(audio, sr, samplerate, audio.shape[0])
    
    return audio


def new_sdr(references, estimates):
    """
    Compute the SDR according to the MDX challenge definition.
    Official implementation from demucs/evaluate.py
    """
    assert references.dim() == 4
    assert estimates.dim() == 4
    delta = 1e-7  # avoid numerical errors
    num = th.sum(th.square(references), dim=(2, 3))
    den = th.sum(th.square(references - estimates), dim=(2, 3))
    num += delta
    den += delta
    scores = 10 * th.log10(num / den)
    return scores


def eval_track_source(reference, estimate, win, hop, compute_traditional_sdr=True):
    """
    Evaluate single source separation using academic standard methods.
    Following demucs official evaluation protocol.
    
    Args:
        reference: Ground truth audio (channels, samples)
        estimate: Estimated audio (channels, samples)
        win: Window size for evaluation
        hop: Hop size for evaluation
        compute_traditional_sdr: Whether to compute traditional SDR (slower)
    
    Returns:
        scores: Dictionary containing SDR scores
    """
    # Prepare tensors following official demucs evaluation protocol
    # Add source dimension: (sources=1, channels, samples)
    ref_3d = reference[None]  # (sources=1, channels, samples)
    est_3d = estimate[None]   # (sources=1, channels, samples)
    
    # Transpose to match official format: (sources, samples, channels)
    ref_transposed = ref_3d.transpose(1, 2).double()
    est_transposed = est_3d.transpose(1, 2).double()
    
    # Add batch dimension and calculate new SDR: (batch=1, sources, samples, channels)
    new_sdr_score = new_sdr(ref_transposed[None], est_transposed[None])[0, 0]
    
    result = {
        'NSDR': float(new_sdr_score),  # New SDR (MDX definition)
    }
    
    if compute_traditional_sdr:
        # Convert to double precision and transpose for museval
        reference_np = reference.transpose(0, 1).double().numpy()  # (samples, channels)
        estimate_np = estimate.transpose(0, 1).double().numpy()    # (samples, channels)
        
        # Add batch dimension for museval
        references = reference_np[None, :, :]  # (1, samples, channels)
        estimates = estimate_np[None, :, :]    # (1, samples, channels)
        
        # Calculate traditional SDR using museval
        scores = museval.metrics.bss_eval(
            references, estimates,
            compute_permutation=False,
            window=win,
            hop=hop,
            framewise_filters=False,
            bsseval_sources_version=False
        )[:-1]  # Remove the last element (permutation)
        
        sdr, isr, sir, sar = scores
        
        result.update({
            'SDR': sdr[0].tolist(),  # Traditional SDR - Remove batch dimension
            'ISR': isr[0].tolist(),
            'SIR': sir[0].tolist(),
            'SAR': sar[0].tolist()
        })
    
    return result


def evaluate_all_tracks_4source(test_dir, model_name="htdemucs", repo_path=None,
                                shifts=1, overlap=0.25, device=None, source_order=None, 
                                compute_traditional_sdr=True):
    """
    Evaluate 4-source separation model on all test tracks.
    
    Args:
        test_dir: Path to test dataset directory (MUSDB18-HQ test set)
        model_name: Model name/hash to use (htdemucs or experiment hash)
        repo_path: Path to model repository (for experiment models)
        shifts: Number of shifts for inference (1)
        overlap: Overlap ratio for inference (0.25)
        device: Device to use ('cuda' or 'cpu')
        source_order: List of source names in model output order, e.g., ['drums', 'bass', 'other', 'vocals']
        compute_traditional_sdr: Whether to compute traditional SDR (slower but more detailed)
    
    Returns:
        results: Dictionary containing evaluation results for all tracks
    """
    
    # Check device availability
    if device is None:
        device = "cuda" if th.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load the model
    if repo_path:
        print(f"Loading model {model_name} from {repo_path}")
        model = get_model(model_name, repo=Path(repo_path))
    else:
        print(f"Loading pretrained model {model_name}")
        model = get_model(model_name)
    
    # Move model to device
    model = model.to(device)
    model.eval()
    
    # Get model parameters
    samplerate = model.samplerate
    win = int(1.0 * samplerate)
    hop = int(1.0 * samplerate)
    
    # Default source order for htdemucs
    if source_order is None:
        source_order = ['drums', 'bass', 'other', 'vocals']
    
    print(f"Source order: {source_order}")
    
    # Find test tracks
    test_path = Path(test_dir)
    if not test_path.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")
    
    track_dirs = [d for d in test_path.iterdir() if d.is_dir()]
    if not track_dirs:
        raise FileNotFoundError(f"No test tracks found in {test_dir}")
    
    print(f"Found {len(track_dirs)} test tracks")
    print("Evaluating tracks...")
    
    all_results = {}
    source_sdr_scores = {source: [] for source in source_order}  # All frames (for overall mean/std)
    source_track_medians = {source: [] for source in source_order}  # Per-track medians (for official metric)
    
    # Evaluate each track
    for track_idx, track_dir in enumerate(track_dirs, 1):
        track_name = track_dir.name
        print(f"\nProcessing track {track_idx}/{len(track_dirs)}: {track_name}")
        
        mixture_path = track_dir / "mixture.wav"
        
        # Check all required reference files
        reference_paths = {}
        missing_sources = []
        for source in source_order:
            ref_path = track_dir / f"{source}.wav"
            if ref_path.exists():
                reference_paths[source] = ref_path
            else:
                missing_sources.append(source)
        
        if not mixture_path.exists():
            print(f"  Missing mixture.wav, skipping...")
            continue
        
        if missing_sources:
            print(f"  Missing reference files: {missing_sources}, skipping...")
            continue
        
        try:
            # Load and process audio following demucs official protocol
            mixture = load_audio(str(mixture_path), samplerate)
            if mixture.dim() == 1:
                mixture = mixture[None]  # Add channel dimension
            
            # Move to device and apply official demucs normalization
            mixture = mixture.to(device)
            
            # Official demucs normalization: use mono mixture statistics
            ref = mixture.mean(dim=0)  # mono mixture: (samples,)
            mixture = (mixture - ref.mean()) / ref.std()
            
            # Apply model for source separation
            with th.no_grad():
                estimates = apply_model(
                    model, mixture[None],  # Add batch dimension
                    shifts=shifts,
                    split=True,
                    overlap=overlap
                )[0]  # Remove batch dimension
            
            # Denormalize estimates using official method
            estimates = estimates * ref.std() + ref.mean()
            
            # Move back to CPU for evaluation
            estimates = estimates.cpu()
            
            # Clear GPU cache to free memory
            if device == "cuda":
                th.cuda.empty_cache()
            
            # Check model output shape
            if estimates.shape[0] != len(source_order):
                print(f"  Warning: Model output has {estimates.shape[0]} sources, expected {len(source_order)}")
                continue
            
            # Load all reference files and stack them
            references_list = []
            for source in source_order:
                ref = load_audio(str(reference_paths[source]), samplerate)
                references_list.append(ref)
            
            # Stack references: (sources, channels, samples)
            references = th.stack(references_list)
            
            # Ensure exact length matching
            target_length = min(estimates.shape[-1], references.shape[-1])
            estimates = estimates[..., :target_length]
            references = references[..., :target_length]
            
            # Evaluate all sources at once (official demucs way)
            # Transpose to (sources, samples, channels) for evaluation
            references_transposed = references.transpose(1, 2).double()
            estimates_transposed = estimates.transpose(1, 2).double()
            
            # Calculate new SDR for all sources
            new_scores = new_sdr(references_transposed.cpu()[None], estimates_transposed.cpu()[None])[0]
            
            # Calculate traditional SDR if requested
            if compute_traditional_sdr:
                references_np = references_transposed.numpy()
                estimates_np = estimates_transposed.numpy()
                
                scores = museval.metrics.bss_eval(
                    references_np, estimates_np,
                    compute_permutation=False,
                    window=win, hop=hop,
                    framewise_filters=False,
                    bsseval_sources_version=False
                )[:-1]
                
                sdr, isr, sir, sar = scores
            
            # Process results for each source
            track_scores = {}
            for i, source in enumerate(source_order):
                source_result = {
                    'NSDR': float(new_scores[i])
                }
                
                if compute_traditional_sdr:
                    source_result.update({
                        'SDR': sdr[i].tolist(),
                        'ISR': isr[i].tolist(),
                        'SIR': sir[i].tolist(),
                        'SAR': sar[i].tolist()
                    })
                    
                    # Calculate statistics
                    sdr_clean = [x for x in sdr[i] if np.isfinite(x)]
                    sdr_clean_mean = np.mean(sdr_clean) if sdr_clean else np.nan
                    sdr_clean_median = np.median(sdr_clean) if sdr_clean else np.nan
                    
                    # Print source result
                    if not np.isnan(sdr_clean_mean):
                        print(f"  {source}: Traditional SDR {sdr_clean_mean:.3f} dB, New SDR {float(new_scores[i]):.3f} dB")
                    else:
                        print(f"  {source}: New SDR {float(new_scores[i]):.3f} dB")
                    
                    # Collect SDR scores for overall statistics
                    source_sdr_scores[source].extend(sdr_clean)  # All frames
                    source_track_medians[source].append(sdr_clean_median)  # Per-track median (official metric)
                else:
                    print(f"  {source}: New SDR {float(new_scores[i]):.3f} dB")
                
                track_scores[source] = source_result
            
            # Store results for this track
            all_results[track_name] = track_scores
            
        except Exception as e:
            print(f"Error evaluating {track_name}: {e}")
            continue
    
    # Calculate overall statistics per source
    print(f"\n=== Overall Results ===")
    print(f"Tracks evaluated: {len(all_results)}")
    
    overall_stats = {}
    for source in source_order:
        if source_sdr_scores[source]:
            # All-frames statistics (for reference)
            mean_sdr = np.mean(source_sdr_scores[source])
            median_sdr = np.median(source_sdr_scores[source])
            std_sdr = np.std(source_sdr_scores[source])
            
            # Official metric: mean of per-track medians
            track_medians = [x for x in source_track_medians[source] if np.isfinite(x)]
            mean_of_medians = np.mean(track_medians) if track_medians else np.nan
            median_of_medians = np.median(track_medians) if track_medians else np.nan
            
            print(f"\n{source.capitalize()}:")
            print(f"  Mean of per-track medians: {mean_of_medians:.3f} dB (official metric)")
            print(f"  Median of per-track medians: {median_of_medians:.3f} dB")
            print(f"  All-frames Mean: {mean_sdr:.3f} dB")
            print(f"  All-frames Median: {median_sdr:.3f} dB")
            print(f"  Std: {std_sdr:.3f} dB")
            
            overall_stats[source] = {
                'mean_of_medians': mean_of_medians,  # Official metric
                'median_of_medians': median_of_medians,
                'mean': mean_sdr,
                'median': median_sdr,
                'std': std_sdr,
                'count': len(source_sdr_scores[source])
            }
    
    # Calculate overall average (official metric: average of mean-of-medians)
    if overall_stats:
        all_mean_of_medians = [stats['mean_of_medians'] for stats in overall_stats.values()]
        all_median_of_medians = [stats['median_of_medians'] for stats in overall_stats.values()]
        all_medians = [stats['median'] for stats in overall_stats.values()]
        all_means = [stats['mean'] for stats in overall_stats.values()]
        
        overall_mean_of_medians = np.mean(all_mean_of_medians)
        overall_median_of_medians = np.mean(all_median_of_medians)
        overall_median = np.mean(all_medians)
        overall_mean = np.mean(all_means)
        
        print(f"\n=== Overall (across all sources) ===")
        print(f"Mean of per-track medians: {overall_mean_of_medians:.3f} dB (official metric)")
        print(f"Median of per-track medians: {overall_median_of_medians:.3f} dB")
        print(f"All-frames Mean: {overall_mean:.3f} dB")
        print(f"All-frames Median: {overall_median:.3f} dB")
        
        overall_stats['overall'] = {
            'mean_of_medians': overall_mean_of_medians,  # Official metric
            'median_of_medians': overall_median_of_medians,
            'mean': overall_mean,
            'median': overall_median,
            'source_mean_of_medians': {source: stats['mean_of_medians'] for source, stats in overall_stats.items() if source != 'overall'},
            'source_medians': {source: stats['median'] for source, stats in overall_stats.items() if source != 'overall'}
        }
        
        return {
            'tracks': all_results,
            'overall_stats': overall_stats,
            'source_sdr_scores': source_sdr_scores
        }
    else:
        logger.error("No valid evaluation results obtained!")
        return None


def main():
    """Main evaluation function."""
    # Configuration
    test_dir = "data/musdb18_hq_test/test"
    
    # Model configuration - modify these to use experiment models
    model_name = "htdemucs"  # For pretrained models: "htdemucs", "htdemucs_ft", etc.
    model_name = "0d31a08f"
    repo_path = None
    repo_path = "./release_models"
    #repo_path = "./release_models"  # For experiment models: set to experiment directory path
    
    # Source order configuration
    # For official htdemucs models:
    source_order = ['drums', 'bass', 'other', 'vocals']
    # For your custom models:
    # source_order = ['vocals', 'drums', 'bass', 'other']
    
    # Example for experiment model:
    # model_name = "your_experiment_hash"  # e.g., "a1b2c3d4"
    # repo_path = "outputs/exp_name"  # path to your experiment directory
    # source_order = ['vocals', 'drums', 'bass', 'other']  # your model's source order
    
    # SDR computation options
    compute_traditional_sdr = True  # Set to False for faster evaluation (only new SDR)
    
    shifts = 1
    overlap = 0.25
    
    print("=== 4-Source Music Separation Evaluation ===")
    print(f"Model: {model_name}")
    if repo_path:
        print(f"Repository: {repo_path}")
    print(f"Test directory: {test_dir}")
    print(f"Source order: {source_order}")
    print(f"Evaluation protocol: Official demucs standard")
    print(f"SDR computation: {'Traditional + New' if compute_traditional_sdr else 'New (MDX 2021) only'}")
    print()
    
    # Run evaluation
    try:
        results = evaluate_all_tracks_4source(
            test_dir=test_dir,
            model_name=model_name,
            repo_path=repo_path,
            shifts=shifts,
            overlap=overlap,
            device="cuda" if th.cuda.is_available() else "cpu",
            source_order=source_order,
            compute_traditional_sdr=compute_traditional_sdr
        )
        
        if results:
            # Save results to file with academic metadata
            import json
            output_file = f"4source_evaluation_{model_name}.json"
            
            # Convert numpy arrays to lists for JSON serialization
            json_results = {
                'evaluation_metadata': {
                    'protocol': 'demucs_4source_standard',
                    'sdr_definition': 'MDX_2021_new_sdr',
                    'normalization': 'mono_mixture_statistics',
                    'model_name': model_name,
                    'source_order': source_order,
                    'shifts': shifts,
                    'overlap': overlap,
                    'traditional_sdr_computed': compute_traditional_sdr
                },
                'overall_stats': results['overall_stats'],
                'tracks': {}
            }
            
            # Process each track's results
            for track_name, track_scores in results['tracks'].items():
                json_results['tracks'][track_name] = {}
                for source, scores in track_scores.items():
                    json_results['tracks'][track_name][source] = {
                        k: v if isinstance(v, list) else [v] if not isinstance(v, (int, float)) else v
                        for k, v in scores.items()
                    }
            
            with open(output_file, 'w') as f:
                json.dump(json_results, f, indent=2)
            
            print(f"\nResults saved to {output_file}")
            
        else:
            print("Evaluation failed!")
            
    except Exception as e:
        print(f"Evaluation error: {e}")
        raise


if __name__ == "__main__":
    main()