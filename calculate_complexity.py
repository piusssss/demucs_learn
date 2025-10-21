import torch
import time
from demucs.htdemucs_p import HTDemucs_p
from demucs.htdemucs_i import HTDemucs_i
from demucs.htdemucs_s import HTDemucs_s
from demucs.htdemucs_d import HTDemucs_d
from demucs.htdemucs_d2 import HTDemucs_d2
from demucs.htdemucs_c import HTDemucs_c
from demucs.htdemucs import HTDemucs
from thop import profile

# Model parameters (these should match your model's __init__ parameters)
samplerate = 44100
speed = True
# Instantiate the model with default parameters (let it use its own default segment)
model = HTDemucs_d2(sources=['vocals', 'drums', 'bass', 'other'], samplerate=samplerate)
segment = model.segment  # Use the model's actual segment parameter

# Move model to GPU if available
if torch.cuda.is_available():
    model = model.cuda()
    print("Using GPU")
else:
    print("Using CPU")

# Create a dummy input tensor
batch_size = 1
audio_channels = 2
length = int(samplerate * segment) # Example length for an 8-second segment
dummy_input = torch.randn(batch_size, audio_channels, length)
if torch.cuda.is_available():
    dummy_input = dummy_input.cuda()
macs, params = profile(model, inputs=(dummy_input,))

if speed:
    from demucs.apply import apply_model
    
    # Real separation speed test
    model.eval()
    test_duration = 180  # 60 seconds test audio
    test_audio = torch.randn(1, 2, int(samplerate * test_duration))
    if torch.cuda.is_available():
        test_audio = test_audio.cuda()
    
    start_time = time.time()
    with torch.no_grad():
        separated = apply_model(model, test_audio, shifts=1, split=True, overlap=0.25, progress=False)
    end_time = time.time()
    
    processing_time = end_time - start_time
    real_time_factor = test_duration / processing_time

print(f"FLOPs (MACs): {macs / 1e9:.2f} G") # Convert to GigaFLOPs
print(f"Parameters: {params / 1e6:.2f} M") # Convert to Million parameters
if speed:
    print(f"Processing time: {processing_time:.2f}s for {test_duration}s audio")
    print(f"Real-time factor: {real_time_factor:.2f}x")