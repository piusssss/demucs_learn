import torch
from demucs.htdemucs_p import HTDemucs_p
from demucs.htdemucs import HTDemucs
from thop import profile

# Model parameters (these should match your model's __init__ parameters)
samplerate = 44100
segment = 8

# Instantiate the model
model = HTDemucs_p(sources=['vocals', 'drums', 'bass', 'other'], samplerate=samplerate, segment=segment)

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

print(f"Model: HTDemucs_p")
print(f"FLOPs (MACs): {macs / 1e9:.2f} G") # Convert to GigaFLOPs
print(f"Parameters: {params / 1e6:.2f} M") # Convert to Million parameters