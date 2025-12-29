import torch
import time
from demucs.htdemucs_2nns import HTDemucs_2nns
from demucs.apply import apply_model
import numpy as np

# Model parameters
samplerate = 44100
cpu = False

# Instantiate the model
model = HTDemucs_2nns(sources=['vocals', 'drums', 'bass', 'other'], samplerate=samplerate)
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
print(f"t_layers: {model.t_layers}")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total_params/1e6:.2f}M")

# Count transformer parameters in detail
if model.unit_transformers is not None:
    print(f"\n{'='*60}")
    print(f"Transformer Parameters (Detailed)")
    print(f"{'='*60}\n")
    
    total_transformer_params = 0
    for t_layer_idx in range(model.t_layers):
        print(f"Transformer Layer {t_layer_idx}:")
        
        # Freq Self-Attention
        freq_self_attn_params = 0
        for res_idx in range(model.num_resolutions):
            params = sum(p.numel() for p in model.unit_transformers[t_layer_idx][0][res_idx].parameters())
            freq_self_attn_params += params
        print(f"  Freq Self-Attn:           {freq_self_attn_params/1e6:.2f}M")
        total_transformer_params += freq_self_attn_params
        
        # Time Self-Attention
        time_self_attn_params = sum(p.numel() for p in model.unit_transformers[t_layer_idx][1].parameters())
        print(f"  Time Self-Attn:           {time_self_attn_params/1e6:.2f}M")
        total_transformer_params += time_self_attn_params
        
        # Time to Freq Cross-Attention
        time_to_freq_params = sum(p.numel() for p in model.unit_transformers[t_layer_idx][2].parameters())
        print(f"  Time→Freq Cross-Attn:     {time_to_freq_params/1e6:.2f}M")
        total_transformer_params += time_to_freq_params
        
        # Freq to Time Cross-Attention
        freq_to_time_params = 0
        for res_idx in range(model.num_resolutions):
            params = sum(p.numel() for p in model.unit_transformers[t_layer_idx][3][res_idx].parameters())
            freq_to_time_params += params
        print(f"  Freq→Time Cross-Attn:     {freq_to_time_params/1e6:.2f}M")
        total_transformer_params += freq_to_time_params
        
        # Final refinement (if last layer)
        if t_layer_idx == model.t_layers - 1:
            freq_final_params = 0
            for res_idx in range(model.num_resolutions):
                params = sum(p.numel() for p in model.unit_transformers[t_layer_idx][4][res_idx].parameters())
                freq_final_params += params
            print(f"  Freq Self-Attn (final):   {freq_final_params/1e6:.2f}M")
            total_transformer_params += freq_final_params
            
            time_final_params = sum(p.numel() for p in model.unit_transformers[t_layer_idx][5].parameters())
            print(f"  Time Self-Attn (final):   {time_final_params/1e6:.2f}M")
            total_transformer_params += time_final_params
        
        print()
    
    print(f"Total Transformer Params:     {total_transformer_params/1e6:.2f}M ({total_transformer_params/total_params*100:.1f}%)")
    print()

# Create test audio for detailed profiling
test_duration_detail = 180
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
    z_list = []
    mag_list = []
    x_list = []
    mean_list = []
    std_list = []
    shapes_list = []
    
    stft_times = []
    for res_idx, (nfft, hop_length) in enumerate(zip(model.nfft_list, model.hop_lengths)):
        start = time.time()
        z = model._spec(mix, nfft=nfft, hop_length=hop_length)
        z_list.append(z)
        
        mag = model._magnitude(z).to(mix.device)
        mag_list.append(mag)
        
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
        stft_times.append(time.time() - start)
    
    for res_idx, t in enumerate(stft_times):
        times[f'1_stft_res_{res_idx}'] = t
    times['1_stft_total'] = sum(stft_times)
    
    # ========== 2. Time branch normalization ==========
    start = time.time()
    xt = mix
    meant = xt.mean(dim=(1, 2), keepdim=True)
    stdt = xt.std(dim=(1, 2), keepdim=True)
    xt = (xt - meant) / (1e-5 + stdt)
    if device == "cuda":
        torch.cuda.synchronize()
    times['2_normalize_time'] = time.time() - start
    
    # ========== 3. Encoder layers ==========
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
        
        # Frequency embedding
        if idx == 0 and model.freq_emb_scale is not None:
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
        times[f'3_encoder_layer_{idx}'] = t
    
    # ========== 4. Transformer ==========
    if model.unit_transformers is not None:
        transformer_layer_times = []
        
        for idx in range(model.t_layers):
            layer_times = {}
            
            # Step 1: Freq Self-Attention
            start = time.time()
            for res_idx in range(model.num_resolutions):
                x_list[res_idx], _ = model.unit_transformers[idx][0][res_idx](x_list[res_idx], xt)
            if device == "cuda":
                torch.cuda.synchronize()
            layer_times['freq_self_attn'] = time.time() - start
            
            # Step 2: Time Self-Attention
            start = time.time()
            _, xt = model.unit_transformers[idx][1](x_list[0], xt)
            if device == "cuda":
                torch.cuda.synchronize()
            layer_times['time_self_attn'] = time.time() - start
            
            # Step 3: Time to ConcatFreq Cross-Attention
            start = time.time()
            _, xt = model.unit_transformers[idx][2](x_list, xt)
            if device == "cuda":
                torch.cuda.synchronize()
            layer_times['time_to_freq_cross_attn'] = time.time() - start
            
            # Step 4: Freq to Time Cross-Attention
            start = time.time()
            for res_idx in range(model.num_resolutions):
                x_list[res_idx], _ = model.unit_transformers[idx][3][res_idx](x_list[res_idx], xt)
            if device == "cuda":
                torch.cuda.synchronize()
            layer_times['freq_to_time_cross_attn'] = time.time() - start
            
            # Final refinement (if last layer)
            if idx == model.t_layers - 1:
                # Step 5: Freq Self-Attention
                start = time.time()
                for res_idx in range(model.num_resolutions):
                    x_list[res_idx], _ = model.unit_transformers[idx][4][res_idx](x_list[res_idx], xt)
                if device == "cuda":
                    torch.cuda.synchronize()
                layer_times['freq_self_attn_final'] = time.time() - start
                
                # Step 6: Time Self-Attention
                start = time.time()
                _, xt = model.unit_transformers[idx][5](x_list[0], xt)
                if device == "cuda":
                    torch.cuda.synchronize()
                layer_times['time_self_attn_final'] = time.time() - start
            
            transformer_layer_times.append(layer_times)
        
        # Store detailed times
        for layer_idx, layer_times_dict in enumerate(transformer_layer_times):
            for key, val in layer_times_dict.items():
                times[f'4_transformer_layer_{layer_idx}_{key}'] = val
        
        total_transformer_time = sum(sum(lt.values()) for lt in transformer_layer_times)
        times['4_transformer_total'] = total_transformer_time
    else:
        times['4_transformer_total'] = 0.0
    
    # ========== 5. Decoder layers ==========
    decoder_times = []
    
    for idx in range(model.depth):
        start = time.time()
        
        # Decode all resolutions
        for res_idx in range(model.num_resolutions):
            skip = saved_list[res_idx].pop(-1)
            target_len = lengths_list[res_idx].pop(-1)
            x_list[res_idx], _ = model.decoders[res_idx][idx](x_list[res_idx], skip, target_len)
        
        # Time domain decoder
        tdec = model.tdecoder[idx]
        length_t = lengths_t.pop(-1)
        skip = saved_t.pop(-1)
        xt, _ = tdec(xt, skip, length_t)
        
        if device == "cuda":
            torch.cuda.synchronize()
        decoder_times.append(time.time() - start)
    
    for idx, t in enumerate(decoder_times):
        times[f'5_decoder_layer_{idx}'] = t
    
    # ========== 6. Post-processing ==========
    start = time.time()
    S = len(model.sources)
    
    for res_idx in range(model.num_resolutions):
        Fq, T = shapes_list[res_idx]
        x_list[res_idx] = x_list[res_idx].view(B, S, -1, Fq, T)
        x_list[res_idx] = x_list[res_idx] * std_list[res_idx][:, None] + mean_list[res_idx][:, None]
    
    if device == "cuda":
        torch.cuda.synchronize()
    times['6_denormalize_freq'] = time.time() - start
    
    # ========== 7. Masking ==========
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
    times['7_masking'] = time.time() - start
    
    # ========== 8. iSTFT ==========
    start = time.time()
    if model.use_train_segment:
        if model.training:
            target_length = length
        else:
            target_length = int(model.segment * model.samplerate)
    else:
        target_length = length
    
    x_time_list = []
    for res_idx in range(model.num_resolutions):
        x_time = model._ispec(zout_list[res_idx], target_length, hop_length=model.hop_lengths[res_idx])
        x_time_list.append(x_time)
    
    if x_is_mps:
        x_time_list = [x.to("mps") for x in x_time_list]
    
    if device == "cuda":
        torch.cuda.synchronize()
    times['8_istft'] = time.time() - start
    
    # ========== 9. Fusion ==========
    start = time.time()
    if model.use_train_segment:
        if model.training:
            xt = xt.reshape(B, S, -1, length)
        else:
            xt = xt.reshape(B, S, -1, target_length)
    else:
        xt = xt.reshape(B, S, -1, length)
    xt = xt * stdt[:, None] + meant[:, None]
    
    # Simple addition (no fusion head in current model)
    x = x_time_list[0] + xt
    
    if device == "cuda":
        torch.cuda.synchronize()
    times['9_fusion'] = time.time() - start

# Print results
print(f"\n{'='*60}")
print(f"Detailed Timing Breakdown")
print(f"{'='*60}\n")

total_time = sum(times.values())

# Define execution order
execution_order = [
    ('1_stft_total', 'Multi-resolution STFT (total)'),
]

for res_idx in range(model.num_resolutions):
    execution_order.append((f'1_stft_res_{res_idx}', f'  └─ Resolution {res_idx} (nfft={model.nfft_list[res_idx]})'))

execution_order.append(('2_normalize_time', 'Normalization (time branch)'))

for idx in range(model.depth):
    execution_order.append((f'3_encoder_layer_{idx}', f'Encoder Layer {idx}'))

execution_order.append(('4_transformer_total', 'Transformer (total)'))

if model.unit_transformers is not None:
    for layer_idx in range(model.t_layers):
        execution_order.append((f'4_transformer_layer_{layer_idx}_freq_self_attn', f'  Layer {layer_idx} - Freq Self-Attn'))
        execution_order.append((f'4_transformer_layer_{layer_idx}_time_self_attn', f'  Layer {layer_idx} - Time Self-Attn'))
        execution_order.append((f'4_transformer_layer_{layer_idx}_time_to_freq_cross_attn', f'  Layer {layer_idx} - Time→Freq Cross-Attn'))
        execution_order.append((f'4_transformer_layer_{layer_idx}_freq_to_time_cross_attn', f'  Layer {layer_idx} - Freq→Time Cross-Attn'))
        if layer_idx == model.t_layers - 1:
            execution_order.append((f'4_transformer_layer_{layer_idx}_freq_self_attn_final', f'  Layer {layer_idx} - Freq Self-Attn (final)'))
            execution_order.append((f'4_transformer_layer_{layer_idx}_time_self_attn_final', f'  Layer {layer_idx} - Time Self-Attn (final)'))

for idx in range(model.depth):
    execution_order.append((f'5_decoder_layer_{idx}', f'Decoder Layer {idx}'))

execution_order.extend([
    ('6_denormalize_freq', 'Denormalization (frequency)'),
    ('7_masking', 'Masking'),
    ('8_istft', 'iSTFT'),
    ('9_fusion', 'Time branch fusion'),
])

print(f"{'Step':<60} {'Time (ms)':<12} {'%':<8}")
print(f"{'-'*80}")

for key, description in execution_order:
    if key in times:
        t = times[key]
        percentage = (t / total_time * 100) if total_time > 0 else 0
        print(f"{description:<60} {t*1000:>10.2f}ms {percentage:>6.1f}%")

print(f"{'-'*80}")
print(f"{'TOTAL':<60} {total_time*1000:>10.2f}ms {100.0:>6.1f}%")

print(f"\n{'='*60}")
print(f"Total time (for {test_duration_detail}s): {total_time*1000:.2f}ms ({total_time:.2f}s)")
print(f"Throughput: {test_duration_detail/total_time:.2f}x realtime")
print(f"{'='*60}\n")
