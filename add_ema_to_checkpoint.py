#!/usr/bin/env python3
"""
Add EMA (Exponential Moving Average) state to an existing checkpoint.
This copies the model weights to create an EMA state, allowing you to
continue training with EMA enabled from a checkpoint that didn't have it.

Usage:
    python add_ema_to_checkpoint.py <checkpoint_path>
    
Example:
    python add_ema_to_checkpoint.py checkpoints/model_e100.pt
"""

import torch
import sys
import os
from pathlib import Path


def add_ema_to_checkpoint(checkpoint_path, output_path=None, backup=True):
    """
    Add EMA state to a checkpoint by copying model weights.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        output_path: Path to save the modified checkpoint (default: overwrite original)
        backup: Whether to create a backup of the original checkpoint
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        return False
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Check if EMA already exists
    if 'ema' in checkpoint:
        print("Warning: Checkpoint already has 'ema' key!")
        print("EMA state already exists. No modification needed.")
        return True
    
    # Check if model state exists (try different keys)
    model_key = None
    if 'model' in checkpoint:
        model_key = 'model'
    elif 'state' in checkpoint:
        model_key = 'state'
    else:
        print("Error: Checkpoint doesn't have 'model' or 'state' key!")
        print(f"Available keys: {list(checkpoint.keys())}")
        return False
    
    print(f"Creating EMA state by copying weights from '{model_key}'...")
    
    # Copy model state to EMA state
    # Use clone() to create independent copies
    checkpoint['ema'] = {
        key: value.clone() 
        for key, value in checkpoint[model_key].items()
    }
    
    print(f"✓ EMA state created with {len(checkpoint['ema'])} parameters")
    
    # Create backup if requested
    if backup:
        # Keep original extension for backup
        backup_path = Path(str(checkpoint_path) + '.backup')
        if not backup_path.exists():
            print(f"Creating backup: {backup_path}")
            torch.save(torch.load(checkpoint_path, map_location='cpu'), backup_path)
            print("✓ Backup created")
        else:
            print(f"Backup already exists: {backup_path}")
    
    # Determine output path
    if output_path is None:
        output_path = checkpoint_path
    else:
        output_path = Path(output_path)
    
    # Save modified checkpoint
    print(f"Saving modified checkpoint: {output_path}")
    torch.save(checkpoint, output_path)
    print("✓ Checkpoint saved successfully")
    
    # Verify
    print("\nVerifying...")
    verify_checkpoint = torch.load(output_path, map_location='cpu')
    if 'ema' in verify_checkpoint:
        print("✓ Verification passed: 'ema' key exists")
        # Use the same model_key we found earlier
        model_key = 'model' if 'model' in verify_checkpoint else 'state'
        print(f"  {model_key.capitalize()} parameters: {len(verify_checkpoint[model_key])}")
        print(f"  EMA parameters: {len(verify_checkpoint['ema'])}")
        return True
    else:
        print("✗ Verification failed: 'ema' key not found!")
        return False


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nError: Please provide checkpoint path")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    
    # Optional: output path
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    success = add_ema_to_checkpoint(
        checkpoint_path, 
        output_path=output_path,
        backup=True
    )
    
    if success:
        print("\n✓ Done! You can now continue training with EMA enabled.")
        sys.exit(0)
    else:
        print("\n✗ Failed to add EMA to checkpoint")
        sys.exit(1)


if __name__ == "__main__":
    main()
