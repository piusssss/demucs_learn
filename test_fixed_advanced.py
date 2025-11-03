#!/usr/bin/env python3
"""
Test script for the fixed HTDemucs Advanced architecture.
"""

import torch
import sys
from pathlib import Path

# Add the demucs directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from demucs.htdemucs_n import HTDemucs_n


def test_fixed_advanced():
    """Test the HTDemucs_n model."""
    print("Testing HTDemucs_n...")
    
    # Create model with conservative parameters for testing
    model = HTDemucs_n(
        sources=['instrumental'],
        audio_channels=2,
        samplerate=44100,
        segment=10,
        # Multi-resolution STFT parameters
        n_ffts=[512, 1024, 2048],  # Smaller for testing
        stft_fusion_method='attention',
        # ResUNet++ parameters
        resunet_base_channels=16,  # Smaller for testing
        resunet_depth=2,           # Smaller for testing
        resunet_use_se=True,
        resunet_use_attention=True,
        # Linear Attention parameters
        linear_attn_layers=2,      # Smaller for testing
        linear_attn_heads=4,
        linear_attn_dim_head=32,
        # Traditional parameters
        channels=32,               # Smaller for testing
        growth=2,
        nfft=2048,                # Smaller for testing
        depth=2,                  # Smaller for testing
        kernel_size=8,
        stride=4,
        time_stride=2,
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created successfully")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Model size: {total_params * 4 / 1024**2:.1f} MB")
    
    # Test forward pass with different lengths
    test_lengths = [44100 * 1, 44100 * 2]  # 1s, 2s
    
    for length in test_lengths:
        print(f"\n  Testing {length/44100:.1f}s audio...")
        
        dummy_input = torch.randn(1, 2, length)
        
        try:
            with torch.no_grad():
                output = model(dummy_input)
            
            print(f"    Input: {dummy_input.shape}")
            print(f"    Output: {output.shape}")
            
            # Check output format
            if output.shape[1] == len(model.sources) and output.shape[2] == model.audio_channels:
                print(f"    ✓ Output format correct")
            else:
                print(f"    ✗ Output format incorrect")
                return False
            
            # Check for NaN/Inf
            if torch.isnan(output).any() or torch.isinf(output).any():
                print(f"    ✗ Output contains NaN/Inf")
                return False
            
            print(f"    ✓ Forward pass successful")
            
        except Exception as e:
            print(f"    ✗ Forward pass failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # Test separation interface
    print(f"\n  Testing separation interface...")
    try:
        audio = torch.randn(2, 44100 * 1)  # 1s stereo
        sources = model.separate(audio, shifts=1)
        print(f"    Separation: {audio.shape} -> {sources.shape}")
        print(f"    ✓ Separation interface works")
    except Exception as e:
        print(f"    ✗ Separation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_components_integration():
    """Test that all three innovations are properly integrated."""
    print("\n  Testing component integration...")
    
    model = HTDemucs_n(
        sources=['instrumental'],
        audio_channels=2,
        n_ffts=[512, 1024],
        channels=16,
        depth=2,
        resunet_base_channels=8,
        resunet_depth=2,
        linear_attn_layers=1,
        linear_attn_heads=2,
    )
    
    # Check that all components exist
    components = {
        'Multi-resolution STFT': hasattr(model, 'multi_stft'),
        'Multi-res Encoder': hasattr(model, 'multi_res_encoder'),
        'ResUNet++': hasattr(model, 'freq_resunet'),
        'Linear Transformer': hasattr(model, 'linear_transformer'),
        'Freq-Time Fusion': hasattr(model, 'freq_time_fusion'),
    }
    
    for name, exists in components.items():
        if exists:
            print(f"    ✓ {name} integrated")
        else:
            print(f"    ✗ {name} missing")
            return False
    
    return True


def main():
    """Run all tests."""
    print("=" * 70)
    print("HTDemucs_n Test Suite")
    print("=" * 70)
    
    # Test component integration
    if not test_components_integration():
        print("\n✗ Component integration test failed")
        return
    
    # Test full model
    if test_fixed_advanced():
        print("\n" + "=" * 70)
        print("✓ All tests passed!")
        print("HTDemucs_n is working correctly.")
        print("\nAll three innovations successfully integrated:")
        print("  • Multi-resolution STFT")
        print("  • ResUNet++ with attention gates")
        print("  • Linear Attention Transformer")
        print("  • Frequency-Time Fusion")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("✗ Tests failed!")
        print("=" * 70)


if __name__ == "__main__":
    main()