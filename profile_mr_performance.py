import torch
import time
from demucs.htdemucs_mr import HTDemucs_mr
from demucs.apply import apply_model
import numpy as np

# Model parameters
samplerate = 44100
cpu = False

# Instantiate the model
model = HTDemucs_mr(sources=['vocals', 'drums', 'bass', 'other'], samplerate=samplerate)
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
print(f"nfft_list: {model.nfft_list}")
print(f"hop_lengths: {model.hop_lengths}")
print(f"num_resolutions: {model.num_resolutions}")
print(f"depth: {model.depth}")
print(f"independent: {model.independent}")
print(f"stride: {model.stride}")
print(f"window_size (resolutions_merge_size): {model.resolutions_merge_size}")
print(f"share: {model.share}")

# Get channels from the first conv layer in encoder
if model.share:
    first_conv = model.encoder[0].conv
else:
    first_conv = model.encoder[0][0].conv

# Get input channels from the first conv layer
if hasattr(first_conv, 'in_channels'):
    channels = first_conv.in_channels
else:
    # Fallback: get from conv weight shape
    channels = first_conv.weight.shape[1]
print(f"input_channels: {channels}")

# Get output channels
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

# Count parameters by layer
print(f"\n{'='*60}")
print(f"Parameters by Layer")
print(f"{'='*60}\n")

# Encoder parameters
encoder_params = []
for idx, encode in enumerate(model.encoder):
    if model.share:
        params = sum(p.numel() for p in encode.parameters())
        encoder_params.append(params)
        print(f"Encoder Layer {idx} (shared): {params/1e6:.2f}M")
    else:
        layer_params = []
        for window_idx, enc in enumerate(encode):
            params = sum(p.numel() for p in enc.parameters())
            layer_params.append(params)
        total_layer_params = sum(layer_params)
        encoder_params.append(total_layer_params)
        print(f"Encoder Layer {idx} ({len(encode)} windows): {total_layer_params/1e6:.2f}M")
        for window_idx, params in enumerate(layer_params):
            print(f"  └─ Window {window_idx}: {params/1e6:.2f}M")

# Transformer parameters
if model.crosstransformer:
    transformer_params = sum(p.numel() for p in model.crosstransformer.parameters())
    print(f"\nTransformer ({t_layers} layers): {transformer_params/1e6:.2f}M")
else:
    transformer_params = 0
    print(f"\nTransformer: 0.00M (disabled)")

# Decoder parameters
decoder_params = []
for idx, decode in enumerate(model.decoder):
    if model.share:
        params = sum(p.numel() for p in decode.parameters())
        decoder_params.append(params)
        print(f"Decoder Layer {idx} (shared): {params/1e6:.2f}M")
    else:
        layer_params = []
        for window_idx, dec in enumerate(decode):
            params = sum(p.numel() for p in dec.parameters())
            layer_params.append(params)
        total_layer_params = sum(layer_params)
        decoder_params.append(total_layer_params)
        print(f"Decoder Layer {idx} ({len(decode)} windows): {total_layer_params/1e6:.2f}M")
        for window_idx, params in enumerate(layer_params):
            print(f"  └─ Window {window_idx}: {params/1e6:.2f}M")

# Frequency embedding
if hasattr(model, 'freq_emb') and model.freq_emb is not None:
    freq_emb_params = sum(p.numel() for p in model.freq_emb.parameters())
    print(f"\nFrequency Embedding: {freq_emb_params/1e6:.2f}M")
else:
    freq_emb_params = 0
    print(f"\nFrequency Embedding: 0.00M")

# Fusion weights
fusion_params = model.final_fusion_weights.numel()
print(f"Final Fusion Weights: {fusion_params/1e6:.4f}M")

# Summary
print(f"\n{'='*60}")
print(f"Parameter Summary")
print(f"{'='*60}\n")
total_encoder_params = sum(encoder_params)
total_decoder_params = sum(decoder_params)
other_params = total_params - total_encoder_params - total_decoder_params - transformer_params - freq_emb_params - fusion_params

print(f"{'Component':<30} {'Parameters':<15} {'Percentage':<10}")
print(f"{'-'*55}")
print(f"{'Encoders (all layers)':<30} {total_encoder_params/1e6:>13.2f}M {total_encoder_params/total_params*100:>8.1f}%")
print(f"{'Decoders (all layers)':<30} {total_decoder_params/1e6:>13.2f}M {total_decoder_params/total_params*100:>8.1f}%")
if transformer_params > 0:
    print(f"{'Transformer':<30} {transformer_params/1e6:>13.2f}M {transformer_params/total_params*100:>8.1f}%")
if freq_emb_params > 0:
    print(f"{'Frequency Embedding':<30} {freq_emb_params/1e6:>13.2f}M {freq_emb_params/total_params*100:>8.1f}%")
print(f"{'Fusion Weights':<30} {fusion_params/1e6:>13.4f}M {fusion_params/total_params*100:>8.1f}%")
if other_params > 0:
    print(f"{'Other':<30} {other_params/1e6:>13.2f}M {other_params/total_params*100:>8.1f}%")
print(f"{'-'*55}")
print(f"{'TOTAL':<30} {total_params/1e6:>13.2f}M {100.0:>8.1f}%")

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
    
    # ========== 1. STFT for all resolutions ==========
    start = time.time()
    z_list = []
    mag_list = []
    for res_idx, (nfft, hop_length) in enumerate(zip(model.nfft_list, model.hop_lengths)):
        z = model._spec(mix, nfft=nfft, hop_length=hop_length)
        mag = model._magnitude(z).to(mix.device)
        z_list.append(z)
        mag_list.append(mag)
    if device == "cuda":
        torch.cuda.synchronize()
    times['1_stft_all'] = time.time() - start
    
    # ========== 2. Normalization ==========
    start = time.time()
    x_list = []
    mean_list = []
    std_list = []
    shapes_list = []
    for mag in mag_list:
        B, C, Fq, T = mag.shape
        shapes_list.append((Fq, T))
        mean = mag.mean(dim=(1, 2, 3), keepdim=True)
        std = mag.std(dim=(1, 2, 3), keepdim=True)
        x = (mag - mean) / (1e-5 + std)
        mean_list.append(mean)
        std_list.append(std)
        x_list.append(x)
    if device == "cuda":
        torch.cuda.synchronize()
    times['2_normalize'] = time.time() - start
    
    # ========== 3. Initial sliding windows ==========
    start = time.time()
    from demucs.htdemucs_mr import create_sliding_windows
    # Check if independent layers exist
    if not model.independent:
        aligned_windows = create_sliding_windows(x_list, model.resolutions_merge_size)
    else:
        aligned_windows = create_sliding_windows(x_list, 1)
    if device == "cuda":
        torch.cuda.synchronize()
    times['3_initial_windows'] = time.time() - start
    
    # ========== 4. Encoder layers ==========
    saved = []
    saved_shapes = []
    lengths = []
    out_shapes = [(B, C*len(model.sources), 1, Fq, T) for Fq, T in shapes_list]
    saved_shapes.append(out_shapes)
    
    encoder_times = []
    window_creation_times = []
    
    for idx, encode in enumerate(model.encoder):
        # Encoding
        start = time.time()
        encoded_windows = []
        encoded_shapes = []
        for window_idx, x in enumerate(aligned_windows):
            lengths.append(x.shape[-1])
            inject = None
            
            if model.share:
                x_encoded = encode(x, inject)
            else:
                x_encoded = encode[window_idx](x, inject)
            
            if idx == 0 and model.freq_emb is not None:
                frs = torch.arange(x_encoded.shape[-2], device=x_encoded.device)
                emb = model.freq_emb(frs).t()[None, :, None, :, None].expand_as(x_encoded)
                x_encoded = x_encoded + model.freq_emb_scale * emb
            
            encoded_windows.append(x_encoded)
            encoded_shapes.append(x_encoded.shape)
        
        if device == "cuda":
            torch.cuda.synchronize()
        encoder_times.append(time.time() - start)
        
        saved_shapes.append(encoded_shapes)
        saved.append(encoded_windows)
        
        # Window creation
        start = time.time()
        if idx >= model.independent - 1:
            aligned_windows = create_sliding_windows(encoded_windows, model.resolutions_merge_size)
        else:
            aligned_windows = create_sliding_windows(encoded_windows, 1)
        if device == "cuda":
            torch.cuda.synchronize()
        window_creation_times.append(time.time() - start)
    
    for idx, t in enumerate(encoder_times):
        times[f'4_encoder_layer_{idx}'] = t
    for idx, t in enumerate(window_creation_times):
        times[f'4_window_creation_{idx}'] = t
    
    # ========== 5. Transformer ==========
    if model.crosstransformer:
        start = time.time()
        aligned_windows[0] = aligned_windows[0].squeeze(2)
        aligned_windows[0] = model.crosstransformer(aligned_windows[0])
        aligned_windows[0] = aligned_windows[0].unsqueeze(2)
        if device == "cuda":
            torch.cuda.synchronize()
        times['5_transformer'] = time.time() - start
    else:
        times['5_transformer'] = 0.0
    
    # ========== 6. Decoder layers ==========
    saved_shapes.pop(-1)
    decoder_times = []
    split_times = []
    
    for idx, decode in enumerate(model.decoder):
        skip_windows = saved.pop(-1)
        layer_shapes = saved_shapes.pop(-1)
        
        # Decoding
        start = time.time()
        decoded_windows = []
        for window_idx, x in enumerate(aligned_windows):
            skip = skip_windows[window_idx]
            length_dec = lengths.pop(-1)
            
            if model.share:
                x_decoded, pre = decode(x, skip, length_dec)
            else:
                x_decoded, pre = decode[window_idx](x, skip, length_dec)
            
            decoded_windows.append(x_decoded)
        if device == "cuda":
            torch.cuda.synchronize()
        decoder_times.append(time.time() - start)
        
        # Split windows
        start = time.time()
        from demucs.htdemucs_mr import split_windows
        if idx < len(model.decoder) - model.independent:
            aligned_windows = split_windows(decoded_windows, model.resolutions_merge_size, layer_shapes)
        else:
            aligned_windows = split_windows(decoded_windows, 1, layer_shapes)
        if idx != len(model.decoder) - 1:
            aligned_windows = [w.unsqueeze(2) for w in aligned_windows]
        if device == "cuda":
            torch.cuda.synchronize()
        split_times.append(time.time() - start)
    
    for idx, t in enumerate(decoder_times):
        times[f'6_decoder_layer_{idx}'] = t
    for idx, t in enumerate(split_times):
        times[f'6_split_windows_{idx}'] = t
    
    # ========== 7. Denormalization ==========
    start = time.time()
    S = len(model.sources)
    x_list = aligned_windows
    for res_idx in range(model.num_resolutions):
        Fq, T = shapes_list[res_idx]
        x_list[res_idx] = x_list[res_idx].view(B, S, -1, Fq, T)
        x_list[res_idx] = x_list[res_idx] * std_list[res_idx][:, None] + mean_list[res_idx][:, None]
    if device == "cuda":
        torch.cuda.synchronize()
    times['7_denormalize'] = time.time() - start
    
    # ========== 8. Masking ==========
    start = time.time()
    zout_list = []
    for res_idx in range(model.num_resolutions):
        zout = model._mask(x_list[res_idx])
        zout_list.append(zout)
    if device == "cuda":
        torch.cuda.synchronize()
    times['8_masking'] = time.time() - start
    
    # ========== 9. iSTFT for all resolutions ==========
    start = time.time()
    x_time_list = []
    for res_idx in range(model.num_resolutions):
        x_time = model._ispec(zout_list[res_idx], length, hop_length=model.hop_lengths[res_idx])
        x_time_list.append(x_time)
    if device == "cuda":
        torch.cuda.synchronize()
    times['9_istft_all'] = time.time() - start
    
    # ========== 10. Multi-resolution fusion ==========
    start = time.time()
    final_weights = torch.nn.functional.softmax(model.final_fusion_weights, dim=1)
    B, S, C, T = x_time_list[0].shape
    x = torch.zeros_like(x_time_list[0])
    for s in range(S):
        for r in range(model.num_resolutions):
            x[:, s] += final_weights[s, r] * x_time_list[r][:, s]
    if device == "cuda":
        torch.cuda.synchronize()
    times['10_fusion'] = time.time() - start

# Print results
print(f"\n{'='*60}")
print(f"Detailed Timing Breakdown (Execution Order)")
print(f"{'='*60}\n")

total_time = sum(times.values())

# Define execution order with clear descriptions
execution_order = [
    ('1_stft_all', 'STFT (all resolutions)'),
    ('2_normalize', 'Normalization'),
    ('3_initial_windows', f'Initial window creation ({model.num_resolutions} res → {"independent" if model.independent else "merged"})'),
]

# Encoder layers
for idx in range(len(model.encoder)):
    execution_order.append((f'4_encoder_layer_{idx}', f'Encoder Layer {idx}'))
    execution_order.append((f'4_window_creation_{idx}', f'  └─ Window creation after Layer {idx}'))

# Transformer
execution_order.append(('5_transformer', 'Transformer (bottleneck)'))

# Decoder layers
for idx in range(len(model.decoder)):
    execution_order.append((f'6_decoder_layer_{idx}', f'Decoder Layer {idx}'))
    execution_order.append((f'6_split_windows_{idx}', f'  └─ Split windows after Layer {idx}'))

# Post-processing
execution_order.extend([
    ('7_denormalize', 'Denormalization'),
    ('8_masking', 'Masking (CAC)'),
    ('9_istft_all', 'iSTFT (all resolutions)'),
    ('10_fusion', 'Multi-resolution fusion'),
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
    'Normalization': sum(t for k, t in times.items() if '2_normalize' in k),
    'Window Operations': sum(t for k, t in times.items() if 'window' in k or '3_initial' in k),
    'Encoder': sum(t for k, t in times.items() if '4_encoder' in k),
    'Transformer': sum(t for k, t in times.items() if '5_transformer' in k),
    'Decoder': sum(t for k, t in times.items() if '6_decoder' in k or '6_split' in k),
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
    'Normalization': sum(t for k, t in times_10s.items() if '2_normalize' in k),
    'Window Operations': sum(t for k, t in times_10s.items() if 'window' in k or '3_initial' in k),
    'Encoder': sum(t for k, t in times_10s.items() if '4_encoder' in k),
    'Transformer': sum(t for k, t in times_10s.items() if '5_transformer' in k),
    'Decoder': sum(t for k, t in times_10s.items() if '6_decoder' in k or '6_split' in k),
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
