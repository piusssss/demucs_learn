#!/usr/bin/env python3
"""
Single track instrumental separation evaluation script using htdemucs_ft model.
Based on evaluate_instrumental.py but modified to evaluate only one track with specific parameters.
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


def eval_track_instrumental(reference, estimate, win, hop):
    """
    Evaluate instrumental separation using traditional SDR calculation.
    
    Args:
        reference: Ground truth instrumental audio (channels, samples)
        estimate: Estimated instrumental audio (channels, samples)
        win: Window size for evaluation
        hop: Hop size for evaluation
    
    Returns:
        scores: Dictionary containing SDR, SIR, ISR, SAR scores
    """
    # Convert to double precision and transpose for museval
    reference = reference.transpose(0, 1).double().numpy()  # (samples, channels)
    estimate = estimate.transpose(0, 1).double().numpy()    # (samples, channels)
    
    # Add batch dimension for museval
    references = reference[None, :, :]  # (1, samples, channels)
    estimates = estimate[None, :, :]    # (1, samples, channels)
    
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
    
    return {
        'SDR': sdr[0].tolist(),  # Remove batch dimension
        'ISR': isr[0].tolist(),
        'SIR': sir[0].tolist(),
        'SAR': sar[0].tolist()
    }


def evaluate_single_track_instrumental(test_dir, model_name="htdemucs_ft", 
                                     shifts=1, overlap=0.25):
    """
    Evaluate instrumental separation model on a single test track.
    
    Args:
        test_dir: Path to test dataset directory
        model_name: Model name to use (htdemucs_ft)
        shifts: Number of shifts for inference (3)
        overlap: Overlap ratio for inference (0.25)
    
    Returns:
        results: Dictionary containing evaluation results
    """
    
    # Load the model
    logger.info(f"Loading model {model_name}")
    model = get_model(model_name)
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
    
    # Use only the first track
    track_dir = track_dirs[0]
    track_name = track_dir.name
    logger.info(f"Evaluating single track: {track_name}")
    
    mixture_path = track_dir / "mixture.wav"
    reference_path = track_dir / "instrumental.wav"
    
    if not mixture_path.exists() or not reference_path.exists():
        raise FileNotFoundError(f"Missing files in {track_name}")
    
    # Load mixture audio
    logger.info("Loading mixture audio...")
    mixture = load_audio(str(mixture_path), samplerate)
    if mixture.dim() == 1:
        mixture = mixture[None]  # Add channel dimension
    
    # Normalize mixture
    ref_std = mixture.std()
    ref_mean = mixture.mean()
    mixture = (mixture - ref_mean) / ref_std
    
    # Apply model to separate sources
    logger.info("Applying model for source separation...")
    with th.no_grad():
        estimates = apply_model(
            model, mixture[None],  # Add batch dimension
            shifts=shifts,
            split=True,
            overlap=overlap
        )[0]  # Remove batch dimension
    
    # Denormalize estimates
    estimates = estimates * ref_std + ref_mean
    
    # For htdemucs model, sources are: ['drums', 'bass', 'other', 'vocals']
    # Instrumental = drums + bass + other (exclude vocals)
    if estimates.shape[0] >= 4:
        # Sum drums, bass, and other to create instrumental
        instrumental_estimate = estimates[0] + estimates[1] + estimates[2]  # drums + bass + other
        logger.info("Created instrumental by summing drums, bass, and other sources")
    else:
        # Fallback for other models
        instrumental_estimate = estimates[0]
        logger.info("Using first source as instrumental (fallback)")
    
    logger.info(f"Model output shape: {estimates.shape}")
    logger.info(f"Instrumental estimate shape: {instrumental_estimate.shape}")
    
    # Load reference instrumental
    logger.info("Loading reference instrumental...")
    reference = load_audio(str(reference_path), samplerate)
    
    # Ensure same length
    min_length = min(instrumental_estimate.shape[-1], reference.shape[-1])
    instrumental_estimate = instrumental_estimate[..., :min_length]
    reference = reference[..., :min_length]
    
    logger.info(f"Audio length for evaluation: {min_length / samplerate:.2f} seconds")
    
    # Evaluate instrumental separation
    logger.info("Evaluating instrumental separation...")
    try:
        scores = eval_track_instrumental(reference, instrumental_estimate, win, hop)
        
        # Calculate statistics
        sdr_mean = np.mean(scores['SDR'])
        sdr_median = np.median(scores['SDR'])
        sdr_std = np.std(scores['SDR'])
        
        logger.info("=== Evaluation Results ===")
        logger.info(f"Track: {track_name}")
        logger.info(f"SDR Mean: {sdr_mean:.3f} dB")
        logger.info(f"SDR Median: {sdr_median:.3f} dB")
        logger.info(f"SDR Std: {sdr_std:.3f} dB")
        logger.info(f"Total frames evaluated: {len(scores['SDR'])}")
        
        # Filter out NaN and Inf values for clean statistics
        sdr_clean = [x for x in scores['SDR'] if np.isfinite(x)]
        if sdr_clean:
            logger.info(f"Clean SDR Mean (no NaN/Inf): {np.mean(sdr_clean):.3f} dB")
            logger.info(f"Valid frames: {len(sdr_clean)}/{len(scores['SDR'])}")
        
        return {
            'track_name': track_name,
            'scores': scores,
            'summary': {
                'sdr_mean': sdr_mean,
                'sdr_median': sdr_median,
                'sdr_std': sdr_std,
                'sdr_clean_mean': np.mean(sdr_clean) if sdr_clean else np.nan,
                'valid_frames': len(sdr_clean),
                'total_frames': len(scores['SDR'])
            }
        }
        
    except Exception as e:
        logger.error(f"Error evaluating {track_name}: {e}")
        raise


def main():
    """Main evaluation function."""
    # Configuration
    test_dir = "data/instrumental_separation/test"
    model_name = "htdemucs_ft"
    shifts = 1
    overlap = 0.25
    
    logger.info("=== Single Track Instrumental Evaluation ===")
    logger.info(f"Model: {model_name}")
    logger.info(f"Shifts: {shifts}")
    logger.info(f"Overlap: {overlap}")
    logger.info(f"Test directory: {test_dir}")
    
    # Run evaluation
    try:
        results = evaluate_single_track_instrumental(
            test_dir=test_dir,
            model_name=model_name,
            shifts=shifts,
            overlap=overlap
        )
        
        if results:
            logger.info("Evaluation completed successfully!")
            
            # Save results to file
            import json
            output_file = f"single_track_evaluation_{results['track_name']}.json"
            
            # Convert numpy arrays to lists for JSON serialization
            json_results = {
                'track_name': results['track_name'],
                'model_config': {
                    'model_name': model_name,
                    'shifts': shifts,
                    'overlap': overlap
                },
                'scores': {k: v if isinstance(v, list) else [v] for k, v in results['scores'].items()},
                'summary': results['summary']
            }
            
            with open(output_file, 'w') as f:
                json.dump(json_results, f, indent=2)
            
            logger.info(f"Results saved to {output_file}")
            
            # Print final summary
            print("\n" + "="*50)
            print("FINAL EVALUATION SUMMARY")
            print("="*50)
            print(f"Track: {results['track_name']}")
            print(f"Model: {model_name} (shifts={shifts}, overlap={overlap})")
            print(f"SDR Mean: {results['summary']['sdr_mean']:.3f} dB")
            print(f"SDR Median: {results['summary']['sdr_median']:.3f} dB")
            print(f"Clean SDR Mean: {results['summary']['sdr_clean_mean']:.3f} dB")
            print(f"Valid Frames: {results['summary']['valid_frames']}/{results['summary']['total_frames']}")
            print("="*50)
            
        else:
            logger.error("Evaluation failed!")
            
    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        raise


if __name__ == "__main__":
    main()