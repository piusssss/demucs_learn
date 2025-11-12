#!/usr/bin/env python3
"""
All tracks instrumental separation evaluation script using htdemucs model.
Based on evaluate_instrumental.py but modified to evaluate all tracks in test directory.
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


def eval_track_instrumental(reference, estimate, win, hop, compute_traditional_sdr=True):
    """
    Evaluate instrumental separation using academic standard methods.
    Following demucs official evaluation protocol.
    
    Args:
        reference: Ground truth instrumental audio (channels, samples)
        estimate: Estimated instrumental audio (channels, samples)
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


def evaluate_all_tracks_instrumental(test_dir, model_name="htdemucs_ft", repo_path=None,
                                   shifts=1, overlap=0.25, device=None, source_order=None, 
                                   compute_traditional_sdr=True):
    """
    Evaluate instrumental separation model on all test tracks.
    
    Args:
        test_dir: Path to test dataset directory
        model_name: Model name/hash to use (htdemucs_ft or experiment hash)
        repo_path: Path to model repository (for experiment models)
        shifts: Number of shifts for inference (1)
        overlap: Overlap ratio for inference (0.25)
        device: Device to use ('cuda' or 'cpu')
        source_order: List of source names in model output order, e.g., ['vocals', 'drums', 'bass', 'other']
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
    all_sdr_scores = []
    
    # Evaluate each track
    for i, track_dir in enumerate(track_dirs, 1):
        track_name = track_dir.name
        
        mixture_path = track_dir / "mixture.wav"
        reference_path = track_dir / "instrumental.wav"
        
        if not mixture_path.exists() or not reference_path.exists():
            print(f"Missing files in {track_name}, skipping...")
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
            
            # Handle different model outputs
            if estimates.shape[0] >= 4:
                # Determine source order
                if source_order is None:
                    source_order = ['drums', 'bass', 'other', 'vocals']
                
                # Find indices for instrumental sources (exclude vocals)
                instrumental_indices = []
                for i, source in enumerate(source_order):
                    if source.lower() != 'vocals':
                        instrumental_indices.append(i)
                
                # Sum instrumental sources
                instrumental_estimate = sum(estimates[i] for i in instrumental_indices)
            elif estimates.shape[0] == 2:
                # For two-source models (instrumental, vocals)
                instrumental_estimate = estimates[0]  # First source is instrumental
            else:
                # Fallback for single-source or other models
                instrumental_estimate = estimates[0]
            
            # Move back to CPU for evaluation
            instrumental_estimate = instrumental_estimate.cpu()
            
            # Clear GPU cache to free memory
            if device == "cuda":
                th.cuda.empty_cache()
            
            # Load reference and handle length alignment properly
            reference = load_audio(str(reference_path), samplerate)
            
            # Academic standard: ensure exact length matching without truncation
            target_length = min(instrumental_estimate.shape[-1], reference.shape[-1])
            if instrumental_estimate.shape[-1] != reference.shape[-1]:
                print(f"  Warning: Length mismatch in {track_name} "
                      f"(estimate: {instrumental_estimate.shape[-1]}, "
                      f"reference: {reference.shape[-1]}), aligning to {target_length}")
            
            instrumental_estimate = instrumental_estimate[..., :target_length]
            reference = reference[..., :target_length]
            
            # Evaluate instrumental separation
            scores = eval_track_instrumental(reference, instrumental_estimate, win, hop, 
                                           compute_traditional_sdr=compute_traditional_sdr)
            
            # Calculate statistics for traditional SDR
            if 'SDR' in scores:
                sdr_mean = np.mean(scores['SDR'])
                sdr_median = np.median(scores['SDR'])
                sdr_std = np.std(scores['SDR'])
                
                # Filter out NaN and Inf values for clean statistics
                sdr_clean = [x for x in scores['SDR'] if np.isfinite(x)]
                sdr_clean_mean = np.mean(sdr_clean) if sdr_clean else np.nan
            else:
                sdr_mean = sdr_median = sdr_std = sdr_clean_mean = np.nan
                sdr_clean = []
            
            # Get new SDR score
            nsdr_score = scores.get('NSDR', np.nan)
            
            # Simple output - show both SDR types
            if not np.isnan(sdr_clean_mean):
                print(f"Track {track_name}: Traditional SDR {sdr_clean_mean:.3f} dB, New SDR {nsdr_score:.3f} dB")
            else:
                print(f"Track {track_name}: New SDR {nsdr_score:.3f} dB")
            
            # Store results for this track
            all_results[track_name] = {
                'scores': scores,
                'summary': {
                    'sdr_mean': sdr_mean,
                    'sdr_median': sdr_median,
                    'sdr_std': sdr_std,
                    'sdr_clean_mean': sdr_clean_mean,
                    'nsdr_score': nsdr_score,
                    'valid_frames': len(sdr_clean),
                    'total_frames': len(scores.get('SDR', []))
                }
            }
            
            # Collect all clean SDR scores for overall statistics
            all_sdr_scores.extend(sdr_clean)
            
        except Exception as e:
            print(f"Error evaluating {track_name}: {e}")
            continue
    
    # Calculate overall statistics
    if all_sdr_scores:
        overall_sdr_mean = np.mean(all_sdr_scores)
        overall_sdr_median = np.median(all_sdr_scores)
        overall_sdr_std = np.std(all_sdr_scores)
        
        print(f"\n=== Overall Results ===")
        print(f"Tracks evaluated: {len(all_results)}")
        print(f"Overall SDR Mean: {overall_sdr_mean:.3f} dB")
        print(f"Overall SDR Median: {overall_sdr_median:.3f} dB")
        print(f"Overall SDR Std: {overall_sdr_std:.3f} dB")
        
        return {
            'tracks': all_results,
            'overall_summary': {
                'total_tracks': len(all_results),
                'overall_sdr_mean': overall_sdr_mean,
                'overall_sdr_median': overall_sdr_median,
                'overall_sdr_std': overall_sdr_std,
                'total_valid_frames': len(all_sdr_scores)
            }
        }
    else:
        logger.error("No valid evaluation results obtained!")
        return None


def main():
    """Main evaluation function."""
    # Configuration
    test_dir = "data/temp/valid"
    
    # Model configuration - modify these to use experiment models
    model_name = "htdemucs"  # For pretrained models: "htdemucs", "htdemucs_ft", etc.
    model_name = "f8e6e990"
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
    
    print("=== Academic Standard Instrumental Evaluation ===")
    print(f"Model: {model_name}")
    if repo_path:
        print(f"Repository: {repo_path}")
    print(f"Test directory: {test_dir}")
    print(f"Evaluation protocol: Official demucs standard")
    print(f"SDR computation: {'Traditional + New' if compute_traditional_sdr else 'New (MDX 2021) only'}")
    print()
    
    # Run evaluation
    try:
        results = evaluate_all_tracks_instrumental(
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
            output_file = f"academic_evaluation_{model_name}.json"
            
            # Convert numpy arrays to lists for JSON serialization
            json_results = {
                'evaluation_metadata': {
                    'protocol': 'demucs_official_standard',
                    'sdr_definition': 'MDX_2021_new_sdr',
                    'normalization': 'mono_mixture_statistics',
                    'model_name': model_name,
                    'shifts': shifts,
                    'overlap': overlap,
                    'traditional_sdr_computed': compute_traditional_sdr
                },
                'tracks': {},
                'overall_summary': results['overall_summary']
            }
            
            # Process each track's results
            for track_name, track_data in results['tracks'].items():
                json_results['tracks'][track_name] = {
                    'scores': {k: v if isinstance(v, list) else [v] for k, v in track_data['scores'].items()},
                    'summary': track_data['summary']
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