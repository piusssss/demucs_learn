import torch
import time
from demucs.htdemucs import HTDemucs
from demucs.apply import apply_model
import numpy as np

# Model parameters
samplerate = 44100
cpu = False

# Instantiate the model
model = HTDemucs(sources=['vocals', 'drums', 'bass', 'other'], samplerate=samplerate)
segment = model.segment

# Move model to GPU if available
if torch.cuda.is_available() and not cpu:
    model = model.cuda()
    device = "cuda"
    print("Using GPU")
else:
    device = "cpu"
    print("Using CPU")

model.eval()

# Print model configuration
print(f"\n{'='*60}")
print(f"Model Configuration")
print(f"{'='*60}\n")
print(f"nfft: {model.nfft}")
print(f"hop_length: {model.hop_length}")
print(f"depth: {model.depth}")
print(f"stride: {model.stride}")
print(f"len(tencoder): {len(model.tencoder)}")
print(f"len(tdecoder): {len(model.tdecoder)}")

# Get channels from encoder
first_conv = model.encoder[0].conv
if hasattr(first_conv, 'in_channels'):
    channels = first_conv.in_channels
else:
    channels = first_conv.weight.shape[1]
print(f"input_channels: {channels}")

if hasattr(first_conv, 'out_channels'):
    out_channels = first_conv.out_channels
else:
    out_channels = first_conv.weight.shape[0]
print(f"output_channels: {out_channels}")

# Get transformer info
if model.crosstransformer:
    t_layers = len(model.crosstransformer.layers)
    print(f"t_layers: {t_layers}")
else:
    print(f"t_layers: 0 (disabled)")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total_params/1e6:.2f}M")

# Create test audio for detailed profiling
test_duration_detail = 10  # 10 seconds for detailed profiling (avoid OOM)
test_audio_detail = torch.randn(1, 2, int(samplerate * test_duration_detail))
if device == "cuda":
    test_audio_detail = test_audio_detail.cuda()

print(f"\n{'='*60}")
print(f"Detailed Profiling - {test_duration_detail}s audio")
print(f"{'='*60}\n")

# Warm up with shorter audio
print("Warming up...")
warmup_audio = torch.randn(1, 2, int(samplerate * 5))
if device == "cuda":
    warmup_audio = warmup_audio.cuda()
with torch.no_grad():
    _ = model(warmup_audio)
if device == "cuda":
    torch.cuda.synchronize()
print("Warm up complete.\n")

# Detailed profiling
times = {}

with torch.no_grad():
    mix = test_audio_detail
    length = mix.shape[-1]
    
    # ========== 1. STFT ==========
    start = time.time()
    z = model._spec(mix)
    mag = model._magnitude(z).to(mix.device)
    if device == "cuda":
        torch.cuda.synchronize()
    times['1_stft'] = time.time() - start
    
    # ========== 2. Normalization (frequency branch) ==========
    start = time.time()
    x = mag
    B, C, Fq, T = x.shape
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    std = x.std(dim=(1, 2, 3), keepdim=True)
    x = (x - mean) / (1e-5 + std)
    if device == "cuda":
        torch.cuda.synchronize()
    times['2_normalize_freq'] = time.time() - start
    
    # ========== 3. Time branch normalization ==========
    start = time.time()
    xt = mix
    meant = xt.mean(dim=(1, 2), keepdim=True)
    stdt = xt.std(dim=(1, 2), keepdim=True)
    xt = (xt - meant) / (1e-5 + stdt)
    if device == "cuda":
        torch.cuda.synchronize()
    times['3_normalize_time'] = time.time() - start
    
    # ========== 4. Encoder layers ==========
    saved = []
    saved_t = []
    lengths = []
    lengths_t = []
    
    encoder_times = []
    tencoder_times = []
    
    for idx, encode in enumerate(model.encoder):
        # Frequency branch
        start = time.time()
        lengths.append(x.shape[-1])
        inject = None
        if idx < len(model.tencoder):
            lengths_t.append(xt.shape[-1])
            tenc = model.tencoder[idx]
            xt = tenc(xt)
            if not tenc.empty:
                saved_t.append(xt)
            else:
                inject = xt
        
        x = encode(x, inject)
        
        if idx == 0 and model.freq_emb is not None:
            frs = torch.arange(x.shape[-2], device=x.device)
            emb = model.freq_emb(frs).t()[None, :, :, None].expand_as(x)
            x = x + model.freq_emb_scale * emb
        
        saved.append(x)
        
        if device == "cuda":
            torch.cuda.synchronize()
        encoder_times.append(time.time() - start)
    
    for idx, t in enumerate(encoder_times):
        times[f'4_encoder_layer_{idx}'] = t
    
    # ========== 5. Transformer ==========
    if model.crosstransformer:
        start = time.time()
        x, xt = model.crosstransformer(x, xt)
        if device == "cuda":
            torch.cuda.synchronize()
        times['5_transformer'] = time.time() - start
    else:
        times['5_transformer'] = 0.0
    
    # ========== 6. Decoder layers ==========
    decoder_times = []
    decoder_freq_times = []
    decoder_time_times = []
    
    for idx, decode in enumerate(model.decoder):
        # Measure frequency branch
        start_freq = time.time()
        skip = saved.pop(-1)
        x, pre = decode(x, skip, lengths.pop(-1))
        if device == "cuda":
            torch.cuda.synchronize()
        freq_time = time.time() - start_freq
        decoder_freq_times.append(freq_time)
        
        # Measure time branch
        start_time = time.time()
        offset = model.depth - len(model.tdecoder)
        if idx >= offset:
            tdec = model.tdecoder[idx - offset]
            length_t = lengths_t.pop(-1)
            if tdec.empty:
                assert pre.shape[2] == 1, pre.shape
                pre = pre[:, :, 0]
                xt, _ = tdec(pre, None, length_t)
            else:
                skip = saved_t.pop(-1)
                xt, _ = tdec(xt, skip, length_t)
        if device == "cuda":
            torch.cuda.synchronize()
        time_time = time.time() - start_time
        decoder_time_times.append(time_time)
        
        decoder_times.append(freq_time + time_time)
    
    for idx, t in enumerate(decoder_times):
        times[f'6_decoder_layer_{idx}'] = t
        times[f'6_decoder_layer_{idx}_freq'] = decoder_freq_times[idx]
        times[f'6_decoder_layer_{idx}_time'] = decoder_time_times[idx]
    
    # ========== 7. Denormalization ==========
    start = time.time()
    S = len(model.sources)
    x = x.view(B, S, -1, Fq, T)
    x = x * std[:, None] + mean[:, None]
    if device == "cuda":
        torch.cuda.synchronize()
    times['7_denormalize_freq'] = time.time() - start
    
    # ========== 8. Masking ==========
    start = time.time()
    x_is_mps = x.device.type == "mps"
    if x_is_mps:
        x = x.cpu()
    zout = model._mask(z, x)
    if device == "cuda":
        torch.cuda.synchronize()
    times['8_masking'] = time.time() - start
    
    # ========== 9. iSTFT ==========
    start = time.time()
    x = model._ispec(zout, length)
    if x_is_mps:
        x = x.to("mps")
    if device == "cuda":
        torch.cuda.synchronize()
    times['9_istft'] = time.time() - start
    
    # ========== 10. Time branch denormalization and fusion ==========
    start = time.time()
    xt = xt.view(B, S, -1, length)
    xt = xt * stdt[:, None] + meant[:, None]
    x = xt + x
    if device == "cuda":
        torch.cuda.synchronize()
    times['10_time_fusion'] = time.time() - start

# Print results
print(f"\n{'='*60}")
print(f"Detailed Timing Breakdown (Execution Order)")
print(f"{'='*60}\n")

total_time = sum(times.values())

# Define execution order with clear descriptions
execution_order = [
    ('1_stft', 'STFT'),
    ('2_normalize_freq', 'Normalization (frequency branch)'),
    ('3_normalize_time', 'Normalization (time branch)'),
]

# Encoder layers
for idx in range(len(model.encoder)):
    execution_order.append((f'4_encoder_layer_{idx}', f'Encoder Layer {idx} (freq + time)'))

# Transformer
execution_order.append(('5_transformer', 'Transformer (cross-attention)'))

# Decoder layers
for idx in range(len(model.decoder)):
    execution_order.append((f'6_decoder_layer_{idx}', f'Decoder Layer {idx} (total)'))
    execution_order.append((f'6_decoder_layer_{idx}_freq', f'  ├─ Frequency branch'))
    execution_order.append((f'6_decoder_layer_{idx}_time', f'  └─ Time branch'))

# Post-processing
execution_order.extend([
    ('7_denormalize_freq', 'Denormalization (frequency)'),
    ('8_masking', 'Masking'),
    ('9_istft', 'iSTFT'),
    ('10_time_fusion', 'Time branch fusion'),
])

print(f"{'Step':<50} {'Time (ms)':<12} {'%':<8}")
print(f"{'-'*70}")

for key, description in execution_order:
    if key in times:
        t = times[key]
        percentage = (t / total_time * 100) if total_time > 0 else 0
        print(f"{description:<50} {t*1000:>10.2f}ms {percentage:>6.1f}%")

print(f"{'-'*70}")
print(f"{'TOTAL':<50} {total_time*1000:>10.2f}ms {100.0:>6.1f}%")

print(f"\n{'='*60}")
print(f"Total time (for {test_duration_detail}s): {total_time*1000:.2f}ms ({total_time:.2f}s)")
print(f"Throughput: {test_duration_detail/total_time:.2f}x realtime")
print(f"Estimated time for 180s: {total_time * 180 / test_duration_detail:.2f}s")
print(f"{'='*60}\n")

# Summary by major categories
print(f"\n{'='*60}")
print(f"Summary by Major Categories")
print(f"{'='*60}\n")

major_categories = {
    'STFT': sum(t for k, t in times.items() if '1_stft' in k),
    'Normalization': sum(t for k, t in times.items() if 'normalize' in k),
    'Encoder': sum(t for k, t in times.items() if '4_encoder' in k),
    'Transformer': sum(t for k, t in times.items() if '5_transformer' in k),
    'Decoder': sum(t for k, t in times.items() if '6_decoder' in k),
    'Post-processing': sum(t for k, t in times.items() if k.startswith('7_') or k.startswith('8_') or k.startswith('9_') or k.startswith('10_'))
}

print(f"{'Category':<25} {'Time (s)':<12} {'Time (ms)':<12} {'Percentage':<12}")
print(f"{'-'*65}")
for cat, t in sorted(major_categories.items(), key=lambda x: x[1], reverse=True):
    print(f"{cat:<25} {t:>10.3f}s {t*1000:>10.1f}ms {t/total_time*100:>10.1f}%")
print(f"{'-'*65}")
print(f"{'TOTAL':<25} {total_time:>10.3f}s {total_time*1000:>10.1f}ms {100.0:>10.1f}%")

# Store 10s times for later scaling
times_10s = times.copy()
total_time_10s = total_time

# Full model test with apply_model (180s)
print(f"\n{'='*60}")
print(f"Full Model Test with apply_model (180s audio)")
print(f"{'='*60}\n")

test_duration_full = 180
test_audio_full = torch.randn(1, 2, int(samplerate * test_duration_full))
if device == "cuda":
    test_audio_full = test_audio_full.cuda()

start_time = time.time()
with torch.no_grad():
    separated = apply_model(model, test_audio_full, shifts=1, split=True, overlap=0.25, progress=False)
if device == "cuda":
    torch.cuda.synchronize()
end_time = time.time()

processing_time = end_time - start_time
real_time_factor = test_duration_full / processing_time

print(f"Processing time: {processing_time:.2f}s for {test_duration_full}s audio")
print(f"Real-time factor: {real_time_factor:.2f}x")
print(f"Average time per second: {processing_time/test_duration_full*1000:.2f}ms/s")

# Comparison
print(f"\n{'='*60}")
print(f"Comparison")
print(f"{'='*60}\n")

# Calculate accurate estimation based on actual segments
segment_length = test_duration_detail  # 10s
overlap = 0.25
stride = segment_length * (1 - overlap)  # 7.5s
num_segments = int(np.ceil((test_duration_full - segment_length) / stride)) + 1

# Estimate merge overhead (empirically ~1-2s for 180s audio)
merge_overhead = 1.0  # seconds

estimated_time_accurate = (total_time_10s * num_segments) + merge_overhead
estimated_time_linear = total_time_10s * 180 / test_duration_detail

print(f"Detailed profiling ({test_duration_detail}s): {total_time_10s:.2f}s")
print(f"\nEstimation methods:")
print(f"  Linear scaling (naive):        {estimated_time_linear:.2f}s")
print(f"  Segment-based (accurate):      {estimated_time_accurate:.2f}s")
print(f"    - Segments to process:       {num_segments}")
print(f"    - Time per segment:          {total_time_10s:.2f}s")
print(f"    - Processing time:           {total_time_10s * num_segments:.2f}s")
print(f"    - Merge overhead:            {merge_overhead:.2f}s")
print(f"\nActual with apply_model (180s):  {processing_time:.2f}s")
print(f"\nAccuracy:")
print(f"  Linear estimate error:         {abs(processing_time - estimated_time_linear):.2f}s ({abs(processing_time - estimated_time_linear)/processing_time*100:.1f}%)")
print(f"  Accurate estimate error:       {abs(processing_time - estimated_time_accurate):.2f}s ({abs(processing_time - estimated_time_accurate)/processing_time*100:.1f}%)")
print(f"\nNote: apply_model uses overlap={overlap} (25%), so it processes {num_segments} segments instead of {int(180/test_duration_detail)}.")

# Extrapolate 10s data to 180s for detailed breakdown
print(f"\n{'='*60}")
print(f"Extrapolated Breakdown for 180s (Segment-based)")
print(f"{'='*60}\n")

scale_factor = num_segments  # Use actual number of segments

print(f"{'Step':<50} {'Time (s)':<12} {'%':<8}")
print(f"{'-'*70}")

for key, description in execution_order:
    if key in times_10s:
        t_10s = times_10s[key]
        t_180s = t_10s * scale_factor
        percentage = (t_10s / total_time_10s * 100) if total_time_10s > 0 else 0
        print(f"{description:<50} {t_180s:>10.2f}s {percentage:>6.1f}%")

print(f"{'-'*70}")
print(f"{'Processing time (estimated)':<50} {total_time_10s * num_segments:>10.2f}s {100.0:>6.1f}%")
print(f"{'Merge overhead (estimated)':<50} {merge_overhead:>10.2f}s")
print(f"{'TOTAL (estimated)':<50} {estimated_time_accurate:>10.2f}s")

# Extrapolated major categories for 180s
print(f"\n{'='*60}")
print(f"Extrapolated Major Categories for 180s")
print(f"{'='*60}\n")

major_categories_10s = {
    'STFT': sum(t for k, t in times_10s.items() if '1_stft' in k),
    'Normalization': sum(t for k, t in times_10s.items() if 'normalize' in k),
    'Encoder': sum(t for k, t in times_10s.items() if '4_encoder' in k),
    'Transformer': sum(t for k, t in times_10s.items() if '5_transformer' in k),
    'Decoder': sum(t for k, t in times_10s.items() if '6_decoder' in k),
    'Post-processing': sum(t for k, t in times_10s.items() if k.startswith('7_') or k.startswith('8_') or k.startswith('9_') or k.startswith('10_'))
}

print(f"{'Category':<25} {'Time (s)':<12} {'Percentage':<12}")
print(f"{'-'*50}")
for cat, t_10s in sorted(major_categories_10s.items(), key=lambda x: x[1], reverse=True):
    t_180s = t_10s * scale_factor
    print(f"{cat:<25} {t_180s:>10.2f}s {t_10s/total_time_10s*100:>10.1f}%")
print(f"{'-'*50}")
processing_time_estimated = total_time_10s * num_segments
print(f"{'Processing (estimated)':<25} {processing_time_estimated:>10.2f}s {100.0:>10.1f}%")
print(f"{'Merge overhead':<25} {merge_overhead:>10.2f}s")
print(f"{'TOTAL (estimated)':<25} {estimated_time_accurate:>10.2f}s")
print(f"{'ACTUAL (apply_model)':<25} {processing_time:>10.2f}s")
print(f"{'DIFFERENCE':<25} {abs(processing_time - estimated_time_accurate):>10.2f}s {abs(processing_time - estimated_time_accurate)/processing_time*100:>10.1f}%")
