#!/usr/bin/env python3
"""
Instrumental separation evaluation script using traditional SDR calculation.
Based on demucs/evaluate.py but specifically for instrumental separation models.
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


def evaluate_instrumental_separation(test_dir, model_name="3fba3fc3", repo_path="./release_models", 
                                   shifts=5, overlap=0.25, workers=4):
    """
    Evaluate instrumental separation model on test dataset.
    
    Args:
        test_dir: Path to test dataset directory
        model_name: Model name/hash to use
        repo_path: Path to model repository
        shifts: Number of shifts for inference
        overlap: Overlap ratio for inference
        workers: Number of worker processes
    
    Returns:
        results: Dictionary containing evaluation results
    """
    
    # Load the model
    logger.info(f"Loading model {model_name} from {repo_path}")
    model = get_model(model_name, repo=Path(repo_path))
    model.eval()
    
    # Get model parameters
    samplerate = model.samplerate
    win = int(1.0 * samplerate)
    hop = int(1.0 * samplerate)
    
    # Find all test tracks
    test_path = Path(test_dir)
    if not test_path.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")
    
    track_dirs = [d for d in test_path.iterdir() if d.is_dir()]
    logger.info(f"Found {len(track_dirs)} test tracks")
    
    # Only process the first track
    if track_dirs:
        track_dirs = track_dirs[:1]
        logger.info(f"Processing only the first track: {track_dirs[0].name}")
    else:
        logger.error("No test tracks found")
        return {}
    
    # Setup progress tracking
    indexes = LogProgress(logger, range(len(track_dirs)), updates=10, name='Eval')
    pendings = []
    
    # Setup process pool
    pool = futures.ProcessPoolExecutor if workers > 0 else DummyPoolExecutor
    
    with pool(workers) as executor:
        for i, track_dir in enumerate(track_dirs):
            track_name = track_dir.name
            mixture_path = track_dir / "mixture.wav"
            reference_path = track_dir / "instrumental.wav"
            
            if not mixture_path.exists() or not reference_path.exists():
                logger.warning(f"Missing files in {track_name}, skipping")
                continue
            
            logger.info(f"Processing track: {track_name}")
            
            # Load mixture audio
            mixture = load_audio(str(mixture_path), samplerate)
            if mixture.dim() == 1:
                mixture = mixture[None]  # Add channel dimension
            
            # Normalize mixture
            ref_std = mixture.std()
            ref_mean = mixture.mean()
            mixture = (mixture - ref_mean) / ref_std
            
            # Apply model to separate instrumental
            with th.no_grad():
                estimates = apply_model(
                    model, mixture[None],  # Add batch dimension
                    shifts=shifts,
                    split=True,
                    overlap=overlap
                )[0]  # Remove batch dimension
            
            # Denormalize estimates
            estimates = estimates * ref_std + ref_mean
            
            # For instrumental separation, we typically want the first output
            # (assuming the model outputs [instrumental, vocals] or similar)
            if estimates.shape[0] > 1:
                instrumental_estimate = estimates[0]  # Take first source
            else:
                instrumental_estimate = estimates[0]
            
            # Load reference instrumental
            reference = load_audio(str(reference_path), samplerate)
            
            # Ensure same length
            min_length = min(instrumental_estimate.shape[-1], reference.shape[-1])
            instrumental_estimate = instrumental_estimate[..., :min_length]
            reference = reference[..., :min_length]
            
            # Submit evaluation task
            pendings.append((
                track_name,
                executor.submit(eval_track_instrumental, reference, instrumental_estimate, win, hop)
            ))
    
    # Collect results
    logger.info("Collecting evaluation results...")
    pendings = LogProgress(logger, pendings, updates=10, name='Eval (BSS)')
    
    all_results = {}
    for track_name, pending in pendings:
        try:
            scores = pending.result()
            all_results[track_name] = scores
            logger.info(f"Track {track_name} - SDR: {np.mean(scores['SDR']):.3f}")
        except Exception as e:
            logger.error(f"Error evaluating {track_name}: {e}")
            continue
    
    # Calculate summary statistics
    if all_results:
        summary = calculate_summary_stats(all_results)
        logger.info("=== Evaluation Summary ===")
        for metric, value in summary.items():
            logger.info(f"{metric}: {value:.3f}")
        
        return {
            'tracks': all_results,
            'summary': summary
        }
    else:
        logger.error("No tracks were successfully evaluated")
        return {}


def calculate_summary_stats(results):
    """Calculate summary statistics from track results."""
    metrics = ['SDR', 'ISR', 'SIR', 'SAR']
    summary = {}
    
    for metric in metrics:
        all_values = []
        medians = []
        
        for track_name, track_results in results.items():
            if metric in track_results:
                values = track_results[metric]
                all_values.extend(values)
                medians.append(np.median(values))
        
        if all_values:
            summary[f'{metric.lower()}_mean'] = np.mean(all_values)
            summary[f'{metric.lower()}_median'] = np.median(all_values)
            summary[f'{metric.lower()}_std'] = np.std(all_values)
            summary[f'{metric.lower()}_median_of_medians'] = np.median(medians)
    
    return summary


def main():
    """Main evaluation function."""
    # Configuration
    test_dir = "data/test/valid"
    model_name = "7061066a"
    repo_path = "./release_models"
    
    # Run evaluation
    try:
        results = evaluate_instrumental_separation(
            test_dir=test_dir,
            model_name=model_name,
            repo_path=repo_path,
            shifts=1,
            overlap=0.25,
            workers=4
        )
        
        if results:
            logger.info("Evaluation completed successfully!")
            
            # Save results to file
            import json
            output_file = "instrumental_evaluation_results.json"
            with open(output_file, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                json_results = {}
                for track, scores in results['tracks'].items():
                    json_results[track] = {k: v if isinstance(v, list) else [v] for k, v in scores.items()}
                
                json.dump({
                    'tracks': json_results,
                    'summary': results['summary']
                }, f, indent=2)
            
            logger.info(f"Results saved to {output_file}")
        else:
            logger.error("Evaluation failed!")
            
    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        raise


if __name__ == "__main__":
    main()