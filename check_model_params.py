#!/usr/bin/env python3
"""
Simple and accurate model parameter checker.
Checks parameters from both model definition and checkpoint file.
"""

import torch
import os
from pathlib import Path
import argparse


def check_model_params(model_name="htdemucs", repo_path=None, checkpoint_path=None):
    """
    Check model parameters from definition and/or checkpoint.
    
    Args:
        model_name: Model name (e.g., "htdemucs", "32_28_50")
        repo_path: Path to model repository (for custom models)
        checkpoint_path: Path to checkpoint file (.th)
    """
    print("="*80)
    print("Model Parameter Checker")
    print("="*80)
    
    # 1. Check from model definition
    print(f"\n1. Model Definition: {model_name}")
    if repo_path:
        print(f"   Repository: {repo_path}")
    
    try:
        from demucs.pretrained import get_model
        
        if repo_path:
            model = get_model(model_name, repo=Path(repo_path))
        else:
            model = get_model(model_name)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"   Total parameters:     {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"   Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
        print(f"   Memory (float32):     {total_params * 4 / 1024**2:.2f} MB")
        
        model_params = total_params
        
    except Exception as e:
        print(f"   Error loading model: {e}")
        model_params = None
    
    # 2. Check from checkpoint file
    if checkpoint_path:
        print(f"\n2. Checkpoint File: {checkpoint_path}")
        
        if not os.path.exists(checkpoint_path):
            print(f"   Error: File not found")
            return
        
        # File size
        file_size = os.path.getsize(checkpoint_path)
        print(f"   File size: {file_size / 1024**2:.2f} MB")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Find state_dict
            if 'state' in checkpoint:
                state_dict = checkpoint['state']
                print(f"   Type: Training checkpoint")
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
                print(f"   Type: Model checkpoint")
            else:
                state_dict = checkpoint
                print(f"   Type: Direct state_dict")
            
            # Count parameters
            ckpt_params = sum(p.numel() for p in state_dict.values())
            print(f"   Parameters: {ckpt_params:,} ({ckpt_params/1e6:.2f}M)")
            
            # Compare if both available
            if model_params is not None:
                print(f"\n3. Comparison:")
                diff = abs(model_params - ckpt_params)
                diff_pct = diff / model_params * 100
                print(f"   Difference: {diff:,} ({diff/1e6:.2f}M, {diff_pct:.2f}%)")
                
                if diff == 0:
                    print(f"   ✅ Perfect match!")
                elif diff_pct < 1:
                    print(f"   ✅ Very close (< 1%)")
                else:
                    print(f"   ⚠️  Mismatch detected")
            
        except Exception as e:
            print(f"   Error loading checkpoint: {e}")
    
    print("\n" + "="*80)


def main():
    """Main function."""
    # Configuration - modify these
    model_name = "htdemucs"  # Model name
    repo_path = None  # Path to custom model repository
    checkpoint_path = None  # Path to checkpoint file
    
    # For custom model from repository:
    model_name = "32_4_100"
    repo_path = "./release_models"
    checkpoint_path = "./release_models\model_bs_roformer_ep_17_sdr_9.6568.ckpt"
    
    # For checkpoint file:
    # model_name = "htdemucs"
    # repo_path = None
    # checkpoint_path = "path/to/checkpoint.th"
    
    print("Model Parameter Checker")
    print(f"Model: {model_name}")
    if repo_path:
        print(f"Repository: {repo_path}")
    if checkpoint_path:
        print(f"Checkpoint: {checkpoint_path}")
    print()
    
    check_model_params(
        model_name=model_name,
        repo_path=repo_path,
        checkpoint_path=checkpoint_path
    )


if __name__ == "__main__":
    main()
