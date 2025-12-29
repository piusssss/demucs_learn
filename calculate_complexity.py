import torch
import time
from demucs.htdemucs_p import HTDemucs_p
from demucs.htdemucs_i import HTDemucs_i
from demucs.htdemucs_s import HTDemucs_s
from demucs.htdemucs_d import HTDemucs_d
from demucs.htdemucs_d2 import HTDemucs_d2
from demucs.htdemucs_d3 import HTDemucs_d3
from demucs.htdemucs_d4 import HTDemucs_d4
from demucs.htdemucs_c import HTDemucs_c
from demucs.htdemucs import HTDemucs
from demucs.hdemucs import HDemucs
from demucs.demucs import Demucs
from demucs.rsdemucs import RSDemucs
from demucs.htdemucs_n import HTDemucs_n
from demucs.htdemucs_nn import HTDemucs_nn
from demucs.htdemucs_2nn import HTDemucs_2nn
from demucs.htdemucs_2nns import HTDemucs_2nns
from demucs.htdemucs_2nnew import HTDemucs_2nnew
from demucs.htdemucs_nc import HTDemucs_nc
from demucs.htdemucs_nf import HTDemucs_nf
from demucs.htdemucs_dnf import HTDemucs_dnf
from demucs.htdemucs_dn import HTDemucs_dn
from demucs.htdemucs_mr import HTDemucs_mr
from demucs.htdemucs_x import HTDemucs_x
from thop import profile

# Model parameters (these should match your model's __init__ parameters) 
samplerate = 44100 
speed = True
cpu = False
# Instantiate the model with default parameters (let it use its own default segment)
model = HTDemucs_2nns(sources=['vocals', 'drums', 'bass', 'other'], samplerate=samplerate)
segment = model.segment  # Use the model's actual segment parameter

# Move model to GPU if available
if torch.cuda.is_available() and cpu!=True:
    model = model.cuda()
    device = "cuda"
    print("Using GPU")
else:
    device = "cpu"
    print("Using CPU")

model.eval()

# Count total parameters using the correct method (same as profile_2nn_performance.py)
total_params = sum(p.numel() for p in model.parameters())

# Create a dummy input tensor for FLOPs calculation
batch_size = 1
audio_channels = 2
length = int(samplerate * segment)
dummy_input = torch.randn(batch_size, audio_channels, length)
if device == "cuda":
    dummy_input = dummy_input.cuda()

# Calculate FLOPs
try:
    macs, _ = profile(model, inputs=(dummy_input,))
except AttributeError as e:
    if "'LayerNorm' object has no attribute 'total_ops'" in str(e):
        print(f"Warning: thop compatibility issue with LayerNorm")
        print("Skipping FLOPs calculation (this doesn't affect model functionality)")
        macs = None
    else:
        raise

if macs is not None:
    print(f"FLOPs (MACs): {macs / 1e9:.2f} G")
else:
    print("FLOPs (MACs): N/A (thop compatibility issue)")
print(f"Parameters: {total_params / 1e6:.2f} M")

# Remove thop hooks before speed test to avoid conflicts
def remove_thop_hooks(module):
    """Recursively remove all thop hooks from the model"""
    if hasattr(module, '_forward_hooks'):
        hooks_to_remove = []
        for hook_id, hook in module._forward_hooks.items():
            # Check if it's a thop hook
            if hasattr(hook, '__module__') and 'thop' in str(hook.__module__):
                hooks_to_remove.append(hook_id)
        for hook_id in hooks_to_remove:
            del module._forward_hooks[hook_id]
    
    # Recursively apply to all children
    for child in module.children():
        remove_thop_hooks(child)

# Clean up thop hooks
remove_thop_hooks(model)

if speed:
    from demucs.apply import apply_model
    
    # Real separation speed test
    test_duration = 180
    test_audio = torch.randn(1, 2, int(samplerate * test_duration))
    if device == "cuda":
        test_audio = test_audio.cuda()
    
    start_time = time.time()
    with torch.no_grad():
        separated = apply_model(model, test_audio, shifts=1, split=True, overlap=0.25, progress=False)
    if device == "cuda":
        torch.cuda.synchronize()
    end_time = time.time()
    
    processing_time = end_time - start_time
    real_time_factor = test_duration / processing_time
    
    print(f"Processing time: {processing_time:.2f}s for {test_duration}s audio")
    print(f"Real-time factor: {real_time_factor:.2f}x")
