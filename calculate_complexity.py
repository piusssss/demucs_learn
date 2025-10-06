import torch
import time
from demucs.htdemucs_p import HTDemucs_p
from demucs.htdemucs_i import HTDemucs_i
from demucs.htdemucs_s import HTDemucs_s
from demucs.htdemucs_d import HTDemucs_d
from demucs.htdemucs_d2 import HTDemucs_d2
from demucs.htdemucs import HTDemucs
from thop import profile

# Model parameters (these should match your model's __init__ parameters)
samplerate = 44100
segment = 10
speed = False
# Instantiate the model with correct parameters
model = HTDemucs_d(sources=['vocals', 'drums', 'bass', 'other'], samplerate=samplerate, segment=segment)

# Create a dummy input tensor
# The input to the model is a mix of audio, typically (batch_size, audio_channels, length)
batch_size = 1
audio_channels = 2
length = int(samplerate * segment) # Example length for an 8-second segment
dummy_input = torch.randn(batch_size, audio_channels, length)

# Calculate FLOPs and parameters
# NOTE: thop might not perfectly capture all operations, especially custom ones.
# It provides a good estimate for standard layers.
macs, params = profile(model, inputs=(dummy_input,))

if speed:
    # Measure inference time
    model.eval()  # Set to evaluation mode
    num_runs = 3  # Number of runs for averaging

    # Warm up GPU
    with torch.no_grad():
        for _ in range(3):
            _ = model(dummy_input)

    # Measure actual inference time
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()

    with torch.no_grad():
        for _ in range(num_runs):
            output = model(dummy_input)
            
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.time()

    avg_inference_time = (end_time - start_time) / num_runs

print(f"FLOPs (MACs): {macs / 1e9:.2f} G") # Convert to GigaFLOPs
print(f"Parameters: {params / 1e6:.2f} M") # Convert to Million parameters
if speed:
    print(f"Average Inference Time: {avg_inference_time:.4f} seconds")
    print(f"Throughput: {1/avg_inference_time:.2f} samples/second")