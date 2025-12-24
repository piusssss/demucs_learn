import torch
import time
from demucs.htdemucs_2nn import HTDemucs_2nn
from demucs.apply import apply_model
import numpy as np

# Model parameters
samplerate = 44100
cpu = False

# Instantiate the model
model = HTDemucs_2nn(sources=['vocals', 'drums', 'bass', 'other'], samplerate=samplerate)
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
print(f"stride: {model.stride}")

# Get channels from the first conv layer in encoder
first_conv = model.encoders[0][0].conv
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

# Count parameters by layer
print(f"\n{'='*60}")
print(f"Parameters by Layer")
print(f"{'='*60}\n")

# Encoder parameters (frequency branch, per resolution)
encoder_params = []
for idx in range(model.depth):
    layer_params = []
    for res_idx in range(model.num_resolutions):
        params = sum(p.numel() for p in model.encoders[res_idx][idx].parameters())
        layer_params.append(params)
    total_layer_params = sum(layer_params)
    encoder_params.append(total_layer_params)
    print(f"Encoder Layer {idx} (freq, {model.num_resolutions} res): {total_layer_params/1e6:.2f}M")

# Time encoder parameters
tencoder_params = []
for idx, tenc in enumerate(model.tencoder):
    params = sum(p.numel() for p in tenc.parameters())
    tencoder_params.append(params)
    print(f"Time Encoder Layer {idx}: {params/1e6:.2f}M")

# Transformer parameters
if model.crosstransformer:
    transformer_params = sum(p.numel() for p in model.crosstransformer.parameters())
    print(f"\nTransformer ({t_layers} layers): {transformer_params/1e6:.2f}M")
else:
    transformer_params = 0
    print(f"\nTransformer: 0.00M (disabled)")

# Decoder parameters (frequency branch, per resolution)
decoder_params = []
for idx in range(model.depth):
    layer_params = []
    for res_idx in range(model.num_resolutions):
        params = sum(p.numel() for p in model.decoders[res_idx][idx].parameters())
        layer_params.append(params)
    total_layer_params = sum(layer_params)
    decoder_params.append(total_layer_params)
    print(f"Decoder Layer {idx} (freq, {model.num_resolutions} res): {total_layer_params/1e6:.2f}M")

# Time decoder parameters
tdecoder_params = []
for idx, tdec in enumerate(model.tdecoder):
    params = sum(p.numel() for p in tdec.parameters())
    tdecoder_params.append(params)
    print(f"Time Decoder Layer {idx}: {params/1e6:.2f}M")

# Frequency embedding (per resolution)
if hasattr(model, 'freq_embeddings') and model.freq_embeddings is not None:
    freq_emb_params = sum(sum(p.numel() for p in emb.parameters()) for emb in model.freq_embeddings)
    print(f"\nFrequency Embeddings ({model.num_resolutions} res): {freq_emb_params/1e6:.2f}M")
else:
    freq_emb_params = 0
    print(f"\nFrequency Embeddings: 0.00M (disabled)")

# Fusion weights
fusion_params = 0
if hasattr(model, 'fusion_weights'):
    fusion_params += model.fusion_weights.numel()
if hasattr(model, 'final_fusion_weights'):
    fusion_params += model.final_fusion_weights.numel()
print(f"Fusion Weights: {fusion_params/1e6:.4f}M")

# Summary
print(f"\n{'='*60}")
print(f"Parameter Summary")
print(f"{'='*60}\n")
print(f"Encoder (freq):      {sum(encoder_params)/1e6:.2f}M ({sum(encoder_params)/total_params*100:.1f}%)")
print(f"Encoder (time):      {sum(tencoder_params)/1e6:.2f}M ({sum(tencoder_params)/total_params*100:.1f}%)")
print(f"Transformer:         {transformer_params/1e6:.2f}M ({transformer_params/total_params*100:.1f}%)")
print(f"Decoder (freq):      {sum(decoder_params)/1e6:.2f}M ({sum(decoder_params)/total_params*100:.1f}%)")
print(f"Decoder (time):      {sum(tdecoder_params)/1e6:.2f}M ({sum(tdecoder_params)/total_params*100:.1f}%)")
print(f"Freq Embeddings:     {freq_emb_params/1e6:.2f}M ({freq_emb_params/total_params*100:.1f}%)")
print(f"Fusion Weights:      {fusion_params/1e6:.4f}M ({fusion_params/total_params*100:.2f}%)")
print(f"{'-'*60}")
print(f"TOTAL:               {total_params/1e6:.2f}M (100.0%)")

# Create test audio for detailed profiling
test_duration_detail = 10  # 10 seconds for detailed profiling
test_audio_detail = torch.randn(1, 2, int(samplerate * test_duration_detail))
if device == "cuda":
    test_audio_detail = test_audio_detail.cuda()

print(f"\n{'='*60}")
print(f"Detailed Profiling - {test_duration_detail}s audio")
print(f"{'='*60}\n")

# Warm up
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
    
    # ========== 1. Multi-resolution STFT ==========
    start = time.time()
    z_list = []
    mag_list = []
    for nfft, hop_length in zip(model.nfft_list, model.hop_lengths):
        z = model._spec(mix, nfft=nfft, hop_length=hop_length)
        z_list.append(z)
        mag = model._magnitude(z).to(mix.device)
        mag_list.append(mag)
    if device == "cuda":
        torch.cuda.synchronize()
    times['1_multi_stft'] = time.time() - start
    
    # ========== 2. Normalization (frequency branches) ==========
    start = time.time()
    x_list = []
    mean_list = []
    std_list = []
    for mag in mag_list:
        mean = mag.mean(dim=(1, 2, 3), keepdim=True)
        std = mag.std(dim=(1, 2, 3), keepdim=True)
        x = (mag - mean) / (1e-5 + std)
        mean_list.append(mean)
        std_list.append(std)
        x_list.append(x)
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
    saved_list = [[] for _ in range(model.num_resolutions)]
    lengths_list = [[] for _ in range(model.num_resolutions)]
    saved_t = []
    lengths_t = []
    
    encoder_times = []
    
    for idx in range(model.depth):
        start = time.time()
        
        # Save lengths
        for res_idx in range(model.num_resolutions):
            lengths_list[res_idx].append(x_list[res_idx].shape[-1])
        
        inject = None
        lengths_t.append(xt.shape[-1])
        tenc = model.tencoder[idx]
        xt = tenc(xt)
        saved_t.append(xt)
        
        # Encode all resolutions
        for res_idx in range(model.num_resolutions):
            x_list[res_idx] = model.encoders[res_idx][idx](x_list[res_idx], inject)
        
        # Add frequency embedding
        if idx == 0 and hasattr(model, 'freq_embeddings') and model.freq_embeddings is not None:
            for res_idx in range(model.num_resolutions):
                frs = torch.arange(x_list[res_idx].shape[-2], device=x_list[res_idx].device)
                emb = model.freq_embeddings[res_idx](frs).t()[None, :, :, None].expand_as(x_list[res_idx])
                x_list[res_idx] = x_list[res_idx] + model.freq_emb_scale * emb
        
        # Save skip connections
        for res_idx in range(model.num_resolutions):
            saved_list[res_idx].append(x_list[res_idx])
        
        if device == "cuda":
            torch.cuda.synchronize()
        encoder_times.append(time.time() - start)
    
    for idx, t in enumerate(encoder_times):
        times[f'4_encoder_layer_{idx}'] = t
    
    # ========== 5. Bottleneck fusion + Transformer ==========
    if model.crosstransformer:
        start = time.time()
        
        # Save pre-fusion states
        pre_fusion_list = [x.clone() for x in x_list]
        
        # Fusion to middle resolution
        mid_idx = model.num_resolutions // 2
        target_shape = x_list[mid_idx].shape
        
        x_aligned_list = []
        for res_idx in range(model.num_resolutions):
            if res_idx == mid_idx:
                x_aligned_list.append(x_list[res_idx])
            else:
                x_aligned = torch.nn.functional.interpolate(
                    x_list[res_idx], size=target_shape[2:], mode='bilinear', align_corners=False
                )
                x_aligned_list.append(x_aligned)
        
        # Uniform fusion
        x = sum(1/model.num_resolutions * x_aligned_list[i] for i in range(model.num_resolutions))
        
        # Transformer
        x, xt = model.crosstransformer(x, xt)
        
        # Split back
        for res_idx in range(model.num_resolutions):
            if res_idx == mid_idx:
                x_split = x
            else:
                x_split = torch.nn.functional.interpolate(
                    x, size=pre_fusion_list[res_idx].shape[2:], mode='bilinear', align_corners=False
                )
            x_list[res_idx] = 1/model.num_resolutions * x_split + (1 - 1/model.num_resolutions) * pre_fusion_list[res_idx]
        
        if device == "cuda":
            torch.cuda.synchronize()
        times['5_bottleneck_transformer'] = time.time() - start
    else:
        times['5_bottleneck_transformer'] = 0.0
    
    # ========== 6. Decoder layers ==========
    decoder_times = []
    
    for idx in range(model.depth):
        start = time.time()
        
        # Decode all resolutions
        for res_idx in range(model.num_resolutions):
            skip = saved_list[res_idx].pop(-1)
            target_len = lengths_list[res_idx].pop(-1)
            x_list[res_idx], _ = model.decoders[res_idx][idx](x_list[res_idx], skip, target_len)
        
        # Time decoder
        tdec = model.tdecoder[idx]
        length_t = lengths_t.pop(-1)
        skip = saved_t.pop(-1)
        xt, _ = tdec(xt, skip, length_t)
        
        if device == "cuda":
            torch.cuda.synchronize()
        decoder_times.append(time.time() - start)
    
    for idx, t in enumerate(decoder_times):
        times[f'6_decoder_layer_{idx}'] = t
    
    # ========== 7. Denormalization ==========
    start = time.time()
    S = len(model.sources)
    for res_idx in range(model.num_resolutions):
        B, C, Fq, T = mag_list[res_idx].shape
        x_list[res_idx] = x_list[res_idx].view(B, S, -1, Fq, T)
        x_list[res_idx] = x_list[res_idx] * std_list[res_idx][:, None] + mean_list[res_idx][:, None]
    if device == "cuda":
        torch.cuda.synchronize()
    times['7_denormalize_freq'] = time.time() - start
    
    # ========== 8. Masking ==========
    start = time.time()
    x_is_mps = x_list[0].device.type == "mps"
    if x_is_mps:
        x_list = [x.cpu() for x in x_list]
    
    zout_list = []
    for res_idx in range(model.num_resolutions):
        zout = model._mask(z_list[res_idx], x_list[res_idx])
        zout_list.append(zout)
    
    if device == "cuda":
        torch.cuda.synchronize()
    times['8_masking'] = time.time() - start
    
    # ========== 9. Multi-resolution iSTFT ==========
    start = time.time()
    x_time_list = []
    for res_idx in range(model.num_resolutions):
        x_time = model._ispec(zout_list[res_idx], length, hop_length=model.hop_lengths[res_idx])
        x_time_list.append(x_time)
    
    if x_is_mps:
        x_time_list = [x.to("mps") for x in x_time_list]
    
    if device == "cuda":
        torch.cuda.synchronize()
    times['9_multi_istft'] = time.time() - start
    
    # ========== 10. Sinkhorn fusion + Time branch ==========
    start = time.time()
    
    # Time branch denormalization
    xt = xt.view(B, S, -1, length)
    xt = xt * stdt[:, None] + meant[:, None]
    
    # Sinkhorn normalization
    final_weights = torch.exp(model.final_fusion_weights)
    for _ in range(20):
        final_weights = final_weights / final_weights.sum(dim=1, keepdim=True)
        final_weights = final_weights / final_weights.sum(dim=0, keepdim=True)
    
    # Source-specific fusion
    x = torch.zeros_like(x_time_list[0])
    for s in range(S):
        for r in range(model.num_resolutions):
            x[:, s] += final_weights[s, r] * x_time_list[r][:, s]
    
    # Add time branch
    x = xt + x
    
    if device == "cuda":
        torch.cuda.synchronize()
    times['10_sinkhorn_fusion'] = time.time() - start

# Print results
print(f"\n{'='*60}")
print(f"Detailed Timing Breakdown (Execution Order)")
print(f"{'='*60}\n")

total_time = sum(times.values())

# Define execution order
execution_order = [
    ('1_multi_stft', 'Multi-resolution STFT'),
    ('2_normalize_freq', 'Normalization (frequency branches)'),
    ('3_normalize_time', 'Normalization (time branch)'),
]

# Encoder layers
for idx in range(model.depth):
    execution_order.append((f'4_encoder_layer_{idx}', f'Encoder Layer {idx} (all resolutions + time)'))

# Transformer
execution_order.append(('5_bottleneck_transformer', 'Bottleneck Fusion + Transformer'))

# Decoder layers
for idx in range(model.depth):
    execution_order.append((f'6_decoder_layer_{idx}', f'Decoder Layer {idx} (all resolutions + time)'))

# Post-processing
execution_order.extend([
    ('7_denormalize_freq', 'Denormalization (all resolutions)'),
    ('8_masking', 'Masking (all resolutions)'),
    ('9_multi_istft', 'Multi-resolution iSTFT'),
    ('10_sinkhorn_fusion', 'Sinkhorn Fusion + Time branch'),
])

print(f"{'Step':<55} {'Time (ms)':<12} {'%':<8}")
print(f"{'-'*75}")

for key, description in execution_order:
    if key in times:
        t = times[key]
        percentage = (t / total_time * 100) if total_time > 0 else 0
        print(f"{description:<55} {t*1000:>10.2f}ms {percentage:>6.1f}%")

print(f"{'-'*75}")
print(f"{'TOTAL':<55} {total_time*1000:>10.2f}ms {100.0:>6.1f}%")

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
    'Multi-STFT': times.get('1_multi_stft', 0),
    'Normalization': times.get('2_normalize_freq', 0) + times.get('3_normalize_time', 0),
    'Encoder': sum(t for k, t in times.items() if '4_encoder' in k),
    'Bottleneck+Transformer': times.get('5_bottleneck_transformer', 0),
    'Decoder': sum(t for k, t in times.items() if '6_decoder' in k),
    'Post-processing': (times.get('7_denormalize_freq', 0) + times.get('8_masking', 0) + 
                       times.get('9_multi_istft', 0) + times.get('10_sinkhorn_fusion', 0))
}

print(f"{'Category':<30} {'Time (s)':<12} {'Time (ms)':<12} {'Percentage':<12}")
print(f"{'-'*70}")
for cat, t in sorted(major_categories.items(), key=lambda x: x[1], reverse=True):
    print(f"{cat:<30} {t:>10.3f}s {t*1000:>10.1f}ms {t/total_time*100:>10.1f}%")
print(f"{'-'*70}")
print(f"{'TOTAL':<30} {total_time:>10.3f}s {total_time*1000:>10.1f}ms {100.0:>10.1f}%")

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

segment_length = test_duration_detail
overlap = 0.25
stride = segment_length * (1 - overlap)
num_segments = int(np.ceil((test_duration_full - segment_length) / stride)) + 1
merge_overhead = 1.0

estimated_time_accurate = (total_time * num_segments) + merge_overhead
estimated_time_linear = total_time * 180 / test_duration_detail

print(f"Detailed profiling ({test_duration_detail}s): {total_time:.2f}s")
print(f"\nEstimation methods:")
print(f"  Linear scaling (naive):        {estimated_time_linear:.2f}s")
print(f"  Segment-based (accurate):      {estimated_time_accurate:.2f}s")
print(f"    - Segments to process:       {num_segments}")
print(f"    - Time per segment:          {total_time:.2f}s")
print(f"    - Processing time:           {total_time * num_segments:.2f}s")
print(f"    - Merge overhead:            {merge_overhead:.2f}s")
print(f"\nActual with apply_model (180s):  {processing_time:.2f}s")
print(f"\nAccuracy:")
print(f"  Linear estimate error:         {abs(processing_time - estimated_time_linear):.2f}s ({abs(processing_time - estimated_time_linear)/processing_time*100:.1f}%)")
print(f"  Accurate estimate error:       {abs(processing_time - estimated_time_accurate):.2f}s ({abs(processing_time - estimated_time_accurate)/processing_time*100:.1f}%)")
