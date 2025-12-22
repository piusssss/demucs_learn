import math

import torch
from torch import nn
from torch.nn import functional as F
from fractions import Fraction
from .transformer_mr import CrossTransformerEncoder
from .demucs3 import rescale_module
from .states import capture_init
from .spec import spectro, ispectro
from .hdemucs3 import pad1d, ScaledEmbedding, HEncLayer, HDecLayer


def create_sliding_windows(x_list, window_size):
    num_resolutions = len(x_list)
    if window_size > num_resolutions:
        return list(x_list)
    
    # If input is 5D [B,C,1,F,T], squeeze to 4D [B,C,F,T]
    if len(x_list[0].shape) == 5:
        x_list = [x.squeeze(2) for x in x_list]
    
    windows = []
    for i in range(num_resolutions - window_size + 1):
        window = x_list[i:i + window_size]
        windows.append(window)
    
    aligned_windows = []
    for window in windows:
        aligned = align_resolutions_to_unified_space(window)
        aligned_windows.append(aligned)
    
    return aligned_windows



def split_windows(x_list, window_size, shapes_list):
    B, C, R, F, T = x_list[0].shape
    overlap_size = window_size - 1
    # print(shapes_list)
    shapes_list = [(s[-2], s[-1]) for s in shapes_list]
    # print(shapes_list)
    # Step 1: Unalign each window to get original resolution sizes
    unaligned_windows = []
    for window_idx, window in enumerate(x_list):
        # Determine which resolutions this window contains
        start_res = window_idx
        end_res = window_idx + window_size
        window_shapes = shapes_list[start_res:end_res]
        
        # Unalign this window
        unaligned = unalign_resolutions_from_unified_space(window, window_shapes)
        unaligned_windows.append(unaligned)
    
    # Step 2: Merge overlaps
    if len(unaligned_windows) == 1:
        return unaligned_windows[0]
    
    split_list = []
    for window_idx, unaligned_list in enumerate(unaligned_windows):
        if window_idx == 0:
            # First window: take all resolutions
            split_list.extend(unaligned_list)
        else:
            # Subsequent windows: merge overlap, then add non-overlap
            for r in range(overlap_size):
                split_list[(r - overlap_size)] = split_list[(r - overlap_size)] + unaligned_list[r]
            for r in range(overlap_size, len(unaligned_list)):
                split_list.append(unaligned_list[r])
    
    return split_list


def align_resolutions_to_unified_space(x_list):
    
    B, C = x_list[0].shape[:2]
    
    max_freq = max(x.shape[2] for x in x_list)
    min_time = min(x.shape[3] for x in x_list)
    
    # Step 1: Pad each resolution to make time divisible by min_time
    padded_list = []
    for x in x_list:
        B, C, Fr, T = x.shape
        num_groups = (T + min_time - 1) // min_time
        target_T = num_groups * min_time
        if target_T > T:
            pad_amount = target_T - T
            x = F.pad(x, (0, pad_amount), mode='constant', value=0)
        padded_list.append(x)
    
    # Step 2: Find max time after padding
    max_time = max(x.shape[3] for x in padded_list)
    
    # Step 3: Align all resolutions
    aligned_list = []
    for x in padded_list:
        B, C, Fr, T = x.shape
        
        # Frequency replication
        freq_repeat = max_freq // Fr
        x_aligned = x.unsqueeze(3).repeat(1, 1, 1, freq_repeat, 1)
        x_aligned = x_aligned.reshape(B, C, max_freq, T)
        
        # Time replication (now guaranteed to be divisible)
        time_repeat = max_time // T
        x_aligned = x_aligned.unsqueeze(4).repeat(1, 1, 1, 1, time_repeat)
        x_aligned = x_aligned.reshape(B, C, max_freq, max_time)
        
        aligned_list.append(x_aligned)
    
    aligned = torch.stack(aligned_list, dim=2)
    return aligned


def unalign_resolutions_from_unified_space(x, shapes_list):
    B, C, R, F_max, T_max = x.shape
    x_list = []
    
    # Calculate min_time for padding calculation
    min_time = min(T for _, T in shapes_list)
    
    for res_idx in range(R):
        
        F_target, T_target = shapes_list[res_idx]
        x_res = x[:, :, res_idx, :, :]  # [B, C, F_max, T_max]
        
        # 1. Frequency: average pooling (remove replication)
        freq_repeat = F_max // F_target
        x_res = x_res.reshape(B, C, F_target, freq_repeat, T_max).mean(dim=3)
        # Now: [B, C, F_target, T_max]
        
        # 2. Time: average pooling then crop
        # Calculate padded time
        num_groups = (T_target + min_time - 1) // min_time
        T_padded = num_groups * min_time
        
        time_repeat = T_max // T_padded
        x_res = x_res.reshape(B, C, F_target, T_padded, time_repeat).mean(dim=4)
        # Now: [B, C, F_target, T_padded]
        
        # Crop to original size
        x_res = x_res[:, :, :, :T_target]
        # Now: [B, C, F_target, T_target]
        
        x_list.append(x_res)
    
    return x_list


class HTDemucs_mr(nn.Module):

    @capture_init
    def __init__(
        self,
        sources,
        # Channels
        audio_channels=2,
        channels=32,
        channels_time=None,
        growth=2,
        # STFT
        nfft_list=[2048,4096,8192],  # Multi-resolution STFT window sizes 
        cac=True,
        # Main structure
        depth=5,
        independent=1,
        rewrite=True,
        # Frequency branch
        freq_emb=0.2,
        emb_scale=10,
        emb_smooth=True,
        # Convolutions
        share=False,
        kernel_size=8,
        stride=4,
        resolutions_merge_size=2,
        context=1,
        context_enc=0,
        # Normalization
        norm_starts=4,
        norm_groups=4,
        # DConv residual branch
        dconv_mode=1,
        dconv_depth=2,
        dconv_comp=8,
        dconv_init=1e-3,
        # Transformer
        t_layers=5,
        t_emb="sin",
        t_hidden_scale=4.0,
        t_heads=8,
        t_dropout=0.0,
        t_max_positions=10000,
        t_norm_in=True,
        t_norm_in_group=False,
        t_group_norm=False,
        t_norm_first=True,
        t_norm_out=True,
        t_max_period=10000.0,
        t_weight_decay=0.0,
        t_lr=None,
        t_layer_scale=True,
        t_gelu=True,
        t_weight_pos_embed=1.0,
        t_sin_random_shift=0,
        t_cape_mean_normalize=True,
        t_cape_augment=True,
        t_cape_glob_loc_scale=[5000.0, 1.0, 1.4],
        t_sparse_self_attn=False,
        t_sparse_cross_attn=False,
        t_mask_type="diag",
        t_mask_random_seed=42,
        t_sparse_attn_window=500,
        t_global_window=100,
        t_sparsity=0.95,
        t_auto_sparsity=False,
        # Weight init
        rescale=0.1,
        # Metadata
        samplerate=44100,
        segment=10,
        use_train_segment=True,
    ):
       
        super().__init__()
        self.cac = cac
        self.audio_channels = audio_channels
        self.sources = sources
        self.depth = depth
        self.samplerate = samplerate
        self.segment = segment
        self.stride = stride
        self.share =share
        self.independent =independent
        self.resolutions_merge_size = resolutions_merge_size
        self.use_train_segment = use_train_segment
        
        # Multi-resolution window sizes
        self.nfft_list = nfft_list
        self.num_resolutions = len(nfft_list)
        self.hop_lengths = [nfft // 2 for nfft in nfft_list]

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        if not self.share:
            # Independent encoders/decoders for each window at each layer
            # Calculate max number of windows at each layer
            self.encoders_per_layer = []
            self.decoders_per_layer = []
            for layer_idx in range(depth):
                if layer_idx < independent:
                    num_windows = self.num_resolutions
                else:
                    num_windows = max(1, self.num_resolutions - self.resolutions_merge_size + 1 - layer_idx+independent)
                self.encoders_per_layer.append(num_windows)
                self.decoders_per_layer.append(num_windows)
        
                # Source-specific fusion weights for final time-domain fusion
        self.final_fusion_weights = nn.Parameter(
            torch.ones(len(self.sources), self.num_resolutions) / self.num_resolutions
        )
        self.register_buffer('final_weight_ema', torch.ones(len(self.sources), self.num_resolutions) / self.num_resolutions)
        self.final_weight_momentum = 0.9  
        
        chin = audio_channels
        chin_z = chin  # number of channels for the freq branch
        if self.cac:
            chin_z *= 2
        chout = channels_time or channels
        chout_z = channels

        for index in range(depth):
            norm = index >= norm_starts
            merge = resolutions_merge_size if index > independent-1 else 1
            context = 0 if index > independent-1 else context
            stri = 2**(merge-1) if index > independent-1 else stride
            ker = 2**(merge-1) if index > independent-1 else kernel_size
            freq = True
            pad = True
            
            kw = {
                "kernel_size": ker,
                "stride": stri,
                'freq': freq,
                "pad": pad,
                "norm": norm,
                "rewrite": rewrite,
                "norm_groups": norm_groups,
                "dconv_kw": {
                    "depth": dconv_depth,
                    "compress": dconv_comp,
                    "init": dconv_init,
                    "gelu": True,
                },
            }
            kw_dec = dict(kw)

            if not self.share:
                # Create independent encoders for each window at this layer
                num_windows = self.encoders_per_layer[index]
                encoder_list = nn.ModuleList()
                for _ in range(num_windows):
                    enc = HEncLayer(
                        chin_z, 
                        chout_z, 
                        dconv=dconv_mode & 1, 
                        context=context_enc,
                        resolutions_merge_size=merge,
                        **kw
                    )
                    encoder_list.append(enc)
                self.encoder.append(encoder_list)
            else:
                # Shared encoder for all windows
                enc = HEncLayer(
                    chin_z, 
                    chout_z, 
                    dconv=dconv_mode & 1, 
                    context=context_enc,
                    resolutions_merge_size=merge,
                    **kw
                )
                self.encoder.append(enc)
            
            
            if index == 0:
                chin = self.audio_channels * len(self.sources)
                chin_z = chin
                if self.cac:
                    chin_z *= 2
            
            if not self.share:
                # Create independent decoders for each window at this layer
                num_windows = self.decoders_per_layer[index]
                decoder_list = nn.ModuleList()
                for _ in range(num_windows):
                    dec = HDecLayer(
                        chout_z,
                        chin_z,
                        dconv=dconv_mode & 2,
                        last=index == 0,
                        context=context,
                        resolutions_merge_size=merge,
                        **kw_dec
                    )
                    decoder_list.append(dec)
                self.decoder.insert(0, decoder_list)
            else:
                # Shared decoder for all windows
                dec = HDecLayer(
                    chout_z,
                    chin_z,
                    dconv=dconv_mode & 2,
                    last=index == 0,
                    context=context,
                    resolutions_merge_size=merge,
                    **kw_dec
                )
                self.decoder.insert(0, dec)

            chin = chout
            chin_z = chout_z
            chout = int(growth * chout)
            chout_z = int(growth * chout_z)

            if index == 0 and freq_emb:
                # Frequency embedding for unified space
                # After first layer and alignment: max_freq = (max_nfft // 2) // stride
                max_nfft = max(self.nfft_list)
                freq_after_first_layer = (max_nfft // 2) // stride
                self.freq_emb = ScaledEmbedding(
                    freq_after_first_layer, chin_z, smooth=emb_smooth, scale=emb_scale
                )
                self.freq_emb_scale = freq_emb

        if rescale:
            rescale_module(self, reference=rescale)

        transformer_channels = channels * growth ** (depth - 1)

        if t_layers > 0:
            self.crosstransformer = CrossTransformerEncoder(
                dim=transformer_channels,
                emb=t_emb,
                hidden_scale=t_hidden_scale,
                num_heads=t_heads,
                num_layers=t_layers,
                dropout=t_dropout,
                max_positions=t_max_positions,
                norm_in=t_norm_in,
                norm_in_group=t_norm_in_group,
                group_norm=t_group_norm,
                norm_first=t_norm_first,
                norm_out=t_norm_out,
                max_period=t_max_period,
                weight_decay=t_weight_decay,
                lr=t_lr,
                layer_scale=t_layer_scale,
                gelu=t_gelu,
                sin_random_shift=t_sin_random_shift,
                weight_pos_embed=t_weight_pos_embed,
                cape_mean_normalize=t_cape_mean_normalize,
                cape_augment=t_cape_augment,
                cape_glob_loc_scale=t_cape_glob_loc_scale,
                sparse_self_attn=t_sparse_self_attn,
                sparse_cross_attn=t_sparse_cross_attn,
                mask_type=t_mask_type,
                mask_random_seed=t_mask_random_seed,
                sparse_attn_window=t_sparse_attn_window,
                global_window=t_global_window,
                sparsity=t_sparsity,
                auto_sparsity=t_auto_sparsity,
            )
        else:
            self.crosstransformer = None
            
    def _spec(self, x, nfft=None, hop_length=None):
        hl = hop_length

        # We re-pad the signal in order to keep the property
        # that the size of the output is exactly the size of the input
        # divided by the stride (here hop_length), when divisible.
        # This is achieved by padding by 1/4th of the kernel size (here nfft).
        # which is not supported by torch.stft.
        # Having all convolution operations follow this convention allow to easily
        # align the time and frequency branches later on.
        assert hl == nfft // 2
        le = int(math.ceil(x.shape[-1] / hl))
        pad = hl
        x = pad1d(x, (pad, pad + le * hl - x.shape[-1]), mode="reflect")

        z = spectro(x, nfft, hl)[..., :-1, :]
        assert z.shape[-1] == le + 3, (z.shape, x.shape, le)
        z = z[..., 1: 1 + le]
        return z

    def _ispec(self, z, length=None, scale=0, hop_length=None):
        hl = hop_length // (2**scale)
        z = F.pad(z, (0, 0, 0, 1))
        z = F.pad(z, (1, 2))
        pad = hl
        le = hl * int(math.ceil(length / hl)) + 2 * pad
        x = ispectro(z, hl, length=le)
        x = x[..., pad: pad + length]
        return x

    def _magnitude(self, z):
        # return the magnitude of the spectrogram, except when cac is True,
        # in which case we just move the complex dimension to the channel one.
        if self.cac:
            B, C, Fr, T = z.shape
            m = torch.view_as_real(z).permute(0, 1, 4, 2, 3)
            m = m.reshape(B, C * 2, Fr, T)
        else:
            m = z.abs()
        return m

    def _mask(self, m):

        B, S, C, Fr, T = m.shape
        out = m.view(B, S, -1, 2, Fr, T).permute(0, 1, 2, 4, 5, 3)
        out = torch.view_as_complex(out.contiguous())
        return out

    def valid_length(self, length: int):
        """
        Return a length that is appropriate for evaluation.
        In our case, always return the training length, unless
        it is smaller than the given length, in which case this
        raises an error.
        """
        if not self.use_train_segment:
            return length
        training_length = int(self.segment * self.samplerate)
        if training_length < length:
            raise ValueError(
                    f"Given length {length} is longer than "
                    f"training length {training_length}")
        return training_length
    
    def forward(self, mix):
        length = mix.shape[-1]
        length_pre_pad = None
        if self.use_train_segment:
            if self.training:
                self.segment = Fraction(mix.shape[-1], self.samplerate)
            else:
                training_length = int(self.segment * self.samplerate)
                if mix.shape[-1] < training_length:
                    length_pre_pad = mix.shape[-1]
                    mix = F.pad(mix, (0, training_length - length_pre_pad))
        # Multi-resolution STFT
        x_list = []
        mean_list = []
        std_list = []
        shapes_list = []
        out_shapes =[]
        
        for res_idx, (nfft, hop_length) in enumerate(zip(self.nfft_list, self.hop_lengths)):
            z = self._spec(mix, nfft=nfft, hop_length=hop_length)
            mag = self._magnitude(z).to(mix.device)
            
            B, C, Fq, T = mag.shape
            # print(f"[STFT] Resolution {res_idx}: {mag.shape}")
            shapes_list.append((Fq, T))
            out_shapes.append((B, C*len(self.sources), 1, Fq, T))
            # Normalize
            mean = mag.mean(dim=(1, 2, 3), keepdim=True)
            std = mag.std(dim=(1, 2, 3), keepdim=True)
            x = (mag - mean) / (1e-5 + std)
            
            mean_list.append(mean)
            std_list.append(std)
            x_list.append(x)
            
        if not self.independent:
            aligned_windows = create_sliding_windows(x_list, self.resolutions_merge_size)
        else:
            aligned_windows = create_sliding_windows(x_list, 1)
        # print(f"[INIT] After create_sliding_windows: {len(aligned_windows)} windows, shapes: {[w.shape for w in aligned_windows]}")
        
        # Multi-resolution skip connections and lengths
        saved = []
        saved_shapes = []
        lengths = []
        saved_shapes.append(out_shapes)
        
        for idx, encode in enumerate(self.encoder):
            # print(f"\n[ENC-{idx}] === START ===")
            # print(f"[ENC-{idx}] Input: {len(aligned_windows)} windows")
            # for i, w in enumerate(aligned_windows):
            #     print(f"[ENC-{idx}] Input window {i}: {w.shape}")
            
            # Process each window separately
            encoded_windows = []
            encoded_shapes = []
            for window_idx, x in enumerate(aligned_windows):
                lengths.append(x.shape[-1])
                inject = None
                
                # Use independent or shared encoder
                if self.share:
                    x_encoded = encode(x, inject)
                else:
                    x_encoded = encode[window_idx](x, inject)
                
                # print(f"[ENC-{idx}] Window {window_idx} after encode: {x_encoded.shape}")
                
                if idx == 0 and self.freq_emb is not None:
                    # add frequency embedding
                    frs = torch.arange(x_encoded.shape[-2], device=x_encoded.device)
                    emb = self.freq_emb(frs).t()[None, :, None, :, None].expand_as(x_encoded)
                    x_encoded = x_encoded + self.freq_emb_scale * emb
                
                encoded_windows.append(x_encoded)
                encoded_shapes.append(x_encoded.shape)
            
            saved_shapes.append(encoded_shapes)
            saved.append(encoded_windows)
            # print(f"[ENC-{idx}] Saved shapes: {encoded_shapes}")
            if idx >= self.independent-1:
                aligned_windows = create_sliding_windows(encoded_windows, self.resolutions_merge_size)
            else:
                aligned_windows = create_sliding_windows(encoded_windows, 1)
            # print(f"[ENC-{idx}] After create_sliding_windows: {len(aligned_windows)} windows")
            # for i, w in enumerate(aligned_windows):
            #     print(f"[ENC-{idx}] Output window {i}: {w.shape}")
            
            
        
        if self.crosstransformer:
            # Input: [B, C, R=1, F, T]
            # Squeeze R dimension for transformer
            aligned_windows[0] = aligned_windows[0].squeeze(2)  # [B, C, F, T]
            
            # Single input self-attention
            aligned_windows[0] = self.crosstransformer(aligned_windows[0])  # [B, C, F, T]
            
            # Restore R dimension
            aligned_windows[0] = aligned_windows[0].unsqueeze(2)  # [B, C, R=1, F, T]
            
        saved_shapes.pop(-1)
        for idx, decode in enumerate(self.decoder):
            # print(f"\n[DEC-{idx}] === START ===")
            skip_windows = saved.pop(-1)
            layer_shapes = saved_shapes.pop(-1)
            # print(f"[DEC-{idx}] Input: {len(aligned_windows)} windows")
            # for i, w in enumerate(aligned_windows):
            #     print(f"[DEC-{idx}] Input window {i}: {w.shape}")
            # print(f"[DEC-{idx}] Skip: {len(skip_windows)} windows")
            # for i, w in enumerate(skip_windows):
            #     print(f"[DEC-{idx}] Skip window {i}: {w.shape}")
            # print(f"[DEC-{idx}] Layer shapes: {layer_shapes}")
            
            decoded_windows = []
            for window_idx, x in enumerate(aligned_windows):
                skip = skip_windows[window_idx]
                length_dec = lengths.pop(-1)
                
                # Use independent or shared decoder
                if self.share:
                    x_decoded, pre = decode(x, skip, length_dec)
                else:
                    x_decoded, pre = decode[window_idx](x, skip, length_dec)
                
                # print(f"[DEC-{idx}] Window {window_idx} after decode: {x_decoded.shape}")
                decoded_windows.append(x_decoded)
            
            if idx < len(self.decoder) - self.independent:
                aligned_windows = split_windows(decoded_windows, self.resolutions_merge_size, layer_shapes)
            else:
                aligned_windows = split_windows(decoded_windows, 1, layer_shapes)
            if idx != len(self.decoder) - 1:
                aligned_windows = [w.unsqueeze(2) for w in aligned_windows]
            # print(f"[DEC-{idx}] After split_windows: {len(aligned_windows)} resolutions")
            # for i, w in enumerate(aligned_windows):
            #     print(f"[DEC-{idx}] Output resolution {i}: {w.shape}")

        assert len(saved) == 0
        assert len(lengths) == 0
        
        S = len(self.sources)
        
        x_list = aligned_windows
        
        # Process each resolution separately
        for res_idx in range(self.num_resolutions):
            Fq, T = shapes_list[res_idx]
            x_list[res_idx] = x_list[res_idx].view(B, S, -1, Fq, T)
            x_list[res_idx] = x_list[res_idx] * std_list[res_idx][:, None] + mean_list[res_idx][:, None]

        # Handle MPS device (doesn't support complex numbers)
        x_is_mps = x_list[0].device.type == "mps"
        if x_is_mps:
            x_list = [x.cpu() for x in x_list]
            
        # Apply masking for each resolution
        zout_list = []
        for res_idx in range(self.num_resolutions):
            zout = self._mask(x_list[res_idx])
            zout_list.append(zout)
        
        # Convert back to time domain for each resolution
        if self.use_train_segment:
            if self.training:
                target_length = length
            else:
                target_length = training_length
        else:
            target_length = length
        
        # print(f"[DEBUG] target_length={target_length}, length={length}, training={self.training}")
            
        # iSTFT for each resolution with corresponding hop_length
        x_time_list = []
        for res_idx in range(self.num_resolutions):
            # print(f"[DEBUG] Before iSTFT res {res_idx}: zout shape={zout_list[res_idx].shape}, hop_length={self.hop_lengths[res_idx]}")
            x_time = self._ispec(zout_list[res_idx], target_length, hop_length=self.hop_lengths[res_idx])
            # print(f"[DEBUG] After iSTFT res {res_idx}: x_time shape={x_time.shape}")
            x_time_list.append(x_time)

        # Back to MPS device
        if x_is_mps:
            x_time_list = [x.to("mps") for x in x_time_list]

        # Source-specific weighted fusion for final output
        final_weights = F.softmax(self.final_fusion_weights, dim=1)  # [S, num_res]
        
        if self.training:
            with torch.no_grad():
                self.final_weight_ema = self.final_weight_momentum * self.final_weight_ema + (1 - self.final_weight_momentum) * final_weights.detach()
            weights = (1 - self.final_weight_momentum) * final_weights + self.final_weight_momentum * self.final_weight_ema.detach()
        else:
            weights = final_weights 
            
        # Initialize output
        B, S, C, T = x_time_list[0].shape
        x = torch.zeros_like(x_time_list[0])
        
        #weights = torch.ones(len(self.sources), self.num_resolutions, device=weights.device) / self.num_resolutions
        # Apply source-specific weights
        for s in range(S):
            for r in range(self.num_resolutions):
                x[:, s] += weights[s, r] * x_time_list[r][:, s]
        
        if length_pre_pad:
            # print(f"[DEBUG] Cropping to length_pre_pad={length_pre_pad}")
            x = x[..., :length_pre_pad]
        # print(f"[DEBUG] Final output shape={x.shape}")
        return x