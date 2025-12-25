import math

from openunmix.filtering import wiener
import torch
from torch import nn
from torch.nn import functional as F
from fractions import Fraction
from einops import rearrange

from .transformer import CrossTransformerEncoder

from .demucs import rescale_module
from .states import capture_init
from .spec import spectro, ispectro
from .hdemucs import pad1d, ScaledEmbedding, HEncLayer, HDecLayer


class HTDemucs_2nn(nn.Module):

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
        nfft_list=[2048, 3072,4096, 6144,8192],  # Multi-resolution STFT window sizes
        wiener_iters=0,
        end_iters=0,
        wiener_residual=False,
        cac=True,
        # Main structure
        depth=4,
        rewrite=True,
        # Frequency branch
        freq_emb=0.2,
        emb_scale=10,
        emb_smooth=True,
        # Convolutions
        kernel_size=8,
        time_stride=2,
        stride=4,
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
        # Before the Transformer
        bottom_channels=0,
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
        # ------ Particuliar parameters
        t_cross_first=False,
        # Weight init
        rescale=0.1,
        # Metadata
        samplerate=44100,
        segment=10,
        use_train_segment=True,
    ):
       
        super().__init__()
        self.cac = cac
        self.wiener_residual = wiener_residual
        self.audio_channels = audio_channels
        self.sources = sources
        self.kernel_size = kernel_size
        self.context = context
        self.stride = stride
        self.depth = depth
        self.bottom_channels = bottom_channels
        self.channels = channels
        self.samplerate = samplerate
        self.segment = segment
        self.use_train_segment = use_train_segment
        
        # Multi-resolution window sizes (configurable)
        self.nfft_list = nfft_list
        self.num_resolutions = len(nfft_list)
        self.hop_lengths = [nfft // 2 for nfft in nfft_list]
        self.wiener_iters = wiener_iters
        self.end_iters = end_iters
        self.freq_emb = None
        assert wiener_iters == end_iters

        # Multi-resolution encoders and decoders (dynamic based on nfft_list)
        self.encoders = nn.ModuleList([nn.ModuleList() for _ in range(self.num_resolutions)])
        self.decoders = nn.ModuleList([nn.ModuleList() for _ in range(self.num_resolutions)])

        self.tencoder = nn.ModuleList()
        self.tdecoder = nn.ModuleList()
        '''
        # Multi-resolution fusion weights with momentum smoothing (for bottleneck)
        self.fusion_weights = nn.Parameter(torch.ones(self.num_resolutions) / self.num_resolutions)
        # EMA buffer for smooth weight updates (similar to optimizer momentum)
        self.register_buffer('weight_ema', torch.ones(self.num_resolutions) / self.num_resolutions)
        self.weight_momentum = 0.9  
        '''
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
            stri = stride
            ker = kernel_size
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
            kwt = dict(kw)
            kwt["freq"] = 0
            kwt["kernel_size"] = kernel_size
            kwt["stride"] = stride
            kwt["pad"] = True
            kw_dec = dict(kw)

            # Create encoders for each resolution (dynamic)
            for res_idx in range(self.num_resolutions):
                enc = HEncLayer(
                    chin_z, chout_z, dconv=dconv_mode & 1, context=context_enc, **kw
                )
                self.encoders[res_idx].append(enc)
            
            if freq:
                tenc = HEncLayer(
                    chin,
                    chout,
                    dconv=dconv_mode & 1,
                    context=context_enc,
                    **kwt
                )
                self.tencoder.append(tenc)
            
            if index == 0:
                chin = self.audio_channels * len(self.sources)
                chin_z = chin
                if self.cac:
                    chin_z *= 2
            
            # Create decoders for each resolution (dynamic)
            for res_idx in range(self.num_resolutions):
                dec = HDecLayer(
                    chout_z,
                    chin_z,
                    dconv=dconv_mode & 2,
                    last=index == 0,
                    context=context,
                    **kw_dec
                )
                self.decoders[res_idx].insert(0, dec)
            
            if freq:
                tdec = HDecLayer(
                    chout,
                    chin,
                    dconv=dconv_mode & 2,
                    last=index == 0,
                    context=context,
                    **kwt
                )
                self.tdecoder.insert(0, tdec)

            chin = chout
            chin_z = chout_z
            chout = int(growth * chout)
            chout_z = int(growth * chout_z)

            if index == 0 and freq_emb:
                # Create frequency embeddings for all resolutions (dynamic)
                self.freq_embeddings = nn.ModuleList()
                for nfft in self.nfft_list:
                    # After first layer: freqs = (nfft // 2) // stride
                    freq_after_first_layer = (nfft // 2) // stride
                    emb = ScaledEmbedding(
                        freq_after_first_layer, chin_z, smooth=emb_smooth, scale=emb_scale
                    )
                    self.freq_embeddings.append(emb)
                self.freq_emb_scale = freq_emb

        if rescale:
            rescale_module(self, reference=rescale)

        transformer_channels = channels * growth ** (depth - 1)
        if bottom_channels:
            self.channel_upsampler = nn.Conv1d(
                transformer_channels, bottom_channels, 1
            )
            self.channel_downsampler = nn.Conv1d(
                bottom_channels, transformer_channels, 1
            )
            self.channel_upsampler_t = nn.Conv1d(
                transformer_channels, bottom_channels, 1
            )
            self.channel_downsampler_t = nn.Conv1d(
                bottom_channels, transformer_channels, 1
            )

            transformer_channels = bottom_channels

        if t_layers > 0:
            self.crosstransformer = CrossTransformerEncoder(
                dim=transformer_channels,
                emb=t_emb,
                hidden_scale=t_hidden_scale,
                num_heads=t_heads,
                num_layers=t_layers,
                cross_first=t_cross_first,
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

    def _mask(self, z, m):
        # Apply masking given the mixture spectrogram `z` and the estimated mask `m`.
        # If `cac` is True, `m` is actually a full spectrogram and `z` is ignored.
        niters = self.wiener_iters
        if self.cac:
            B, S, C, Fr, T = m.shape
            out = m.view(B, S, -1, 2, Fr, T).permute(0, 1, 2, 4, 5, 3)
            out = torch.view_as_complex(out.contiguous())
            return out
        if self.training:
            niters = self.end_iters
        if niters < 0:
            z = z[:, None]
            return z / (1e-8 + z.abs()) * m
        else:
            return self._wiener(m, z, niters)

    def _wiener(self, mag_out, mix_stft, niters):
        # apply wiener filtering from OpenUnmix.
        init = mix_stft.dtype
        wiener_win_len = 300
        residual = self.wiener_residual

        B, S, C, Fq, T = mag_out.shape
        mag_out = mag_out.permute(0, 4, 3, 2, 1)
        mix_stft = torch.view_as_real(mix_stft.permute(0, 3, 2, 1))

        outs = []
        for sample in range(B):
            pos = 0
            out = []
            for pos in range(0, T, wiener_win_len):
                frame = slice(pos, pos + wiener_win_len)
                z_out = wiener(
                    mag_out[sample, frame],
                    mix_stft[sample, frame],
                    niters,
                    residual=residual,
                )
                out.append(z_out.transpose(-1, -2))
            outs.append(torch.cat(out, dim=0))
        out = torch.view_as_complex(torch.stack(outs, 0))
        out = out.permute(0, 4, 3, 2, 1).contiguous()
        if residual:
            out = out[:, :-1]
        assert list(out.shape) == [B, S, C, Fq, T]
        return out.to(init)

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
        # Multi-resolution STFT (dynamic)
        z_list = []
        mag_list = []
        x_list = []
        mean_list = []
        std_list = []
        shapes_list = []
        
        for nfft, hop_length in zip(self.nfft_list, self.hop_lengths):
            z = self._spec(mix, nfft=nfft, hop_length=hop_length)
            z_list.append(z)
            
            mag = self._magnitude(z).to(mix.device)
            mag_list.append(mag)
            
            B, C, Fq, T = mag.shape
            shapes_list.append((Fq, T))
            
            # Normalize
            mean = mag.mean(dim=(1, 2, 3), keepdim=True)
            std = mag.std(dim=(1, 2, 3), keepdim=True)
            x = (mag - mean) / (1e-5 + std)
            
            mean_list.append(mean)
            std_list.append(std)
            x_list.append(x)

        # Prepare the time branch input.
        xt = mix
        meant = xt.mean(dim=(1, 2), keepdim=True)
        stdt = xt.std(dim=(1, 2), keepdim=True)
        xt = (xt - meant) / (1e-5 + stdt)

        # Multi-resolution skip connections and lengths (dynamic)
        saved_list = [[] for _ in range(self.num_resolutions)]
        lengths_list = [[] for _ in range(self.num_resolutions)]
        saved_t = []  # skip connections, time branch (shared)
        lengths_t = []  # saved lengths for time branch (shared)
        
        for idx in range(self.depth):
            # Save lengths for each resolution
            for res_idx in range(self.num_resolutions):
                lengths_list[res_idx].append(x_list[res_idx].shape[-1])
            
            inject = None
            lengths_t.append(xt.shape[-1])
            tenc = self.tencoder[idx]
            xt = tenc(xt)
            saved_t.append(xt)
            
            # Encode all resolutions in parallel
            for res_idx in range(self.num_resolutions):
                x_list[res_idx] = self.encoders[res_idx][idx](x_list[res_idx], inject)
            # Add frequency embedding after first layer
            if idx == 0 and self.freq_emb_scale is not None:
                for res_idx in range(self.num_resolutions):
                    frs = torch.arange(x_list[res_idx].shape[-2], device=x_list[res_idx].device)
                    emb = self.freq_embeddings[res_idx](frs).t()[None, :, :, None].expand_as(x_list[res_idx])
                    x_list[res_idx] = x_list[res_idx] + self.freq_emb_scale * emb

            # Save skip connections for each resolution
            for res_idx in range(self.num_resolutions):
                saved_list[res_idx].append(x_list[res_idx])
        '''    
        # Apply softmax to raw weights
        raw_weights = F.softmax(self.fusion_weights, dim=0)
        
        # Apply momentum smoothing during training (like optimizer momentum)
        if self.training:
            with torch.no_grad():
                self.weight_ema = self.weight_momentum * self.weight_ema + (1 - self.weight_momentum) * raw_weights.detach()
            weights = (1 - self.weight_momentum) * raw_weights + self.weight_momentum * self.weight_ema.detach()
        else:
            weights = raw_weights 
        '''        
        if self.crosstransformer:
            # Save pre-fusion states for residual connections
            pre_fusion_list = [x.clone() for x in x_list]
            
            # Adaptive fusion to middle resolution (as target)
            mid_idx = self.num_resolutions // 2
            target_shape = x_list[mid_idx].shape  # [B, C, F_mid, T]
            
            # Interpolate all resolutions to match target shape
            x_aligned_list = []
            for res_idx in range(self.num_resolutions):
                if res_idx == mid_idx:
                    x_aligned_list.append(x_list[res_idx])
                else:
                    x_aligned = F.interpolate(x_list[res_idx], size=target_shape[2:], mode='bilinear', align_corners=False)
                    x_aligned_list.append(x_aligned)
            
            #weights = torch.ones(self.num_resolutions, device=weights.device) / self.num_resolutions
            # Weighted fusion
            x = sum(1/self.num_resolutions * x_aligned_list[i] for i in range(self.num_resolutions))
            
            if self.bottom_channels:
                b, c, f, t = x.shape
                x = rearrange(x, "b c f t-> b c (f t)")
                x = self.channel_upsampler(x)
                x = rearrange(x, "b c (f t)-> b c f t", f=f)
                xt = self.channel_upsampler_t(xt)

            x, xt = self.crosstransformer(x, xt)

            if self.bottom_channels:
                x = rearrange(x, "b c f t-> b c (f t)")
                x = self.channel_downsampler(x)
                x = rearrange(x, "b c (f t)-> b c f t", f=f)
                xt = self.channel_downsampler_t(xt)
            
            # Split the fused feature back to all resolutions
            for res_idx in range(self.num_resolutions):
                if res_idx == mid_idx:
                    x_split = x
                else:
                    x_split = F.interpolate(x, size=pre_fusion_list[res_idx].shape[2:], mode='bilinear', align_corners=False)
                
                # Use fusion weights for residual mixing
                x_list[res_idx] = 1/self.num_resolutions * x_split + (1 - 1/self.num_resolutions) * pre_fusion_list[res_idx]

        for idx in range(self.depth):
            # Decode all resolutions in parallel
            for res_idx in range(self.num_resolutions):
                skip = saved_list[res_idx].pop(-1)
                target_len = lengths_list[res_idx].pop(-1)
                x_list[res_idx], _ = self.decoders[res_idx][idx](x_list[res_idx], skip, target_len)

            # Time domain decoder (shared across resolutions)
            tdec = self.tdecoder[idx]
            length_t = lengths_t.pop(-1)
            skip = saved_t.pop(-1)
            xt, _ = tdec(xt, skip, length_t)
            
        # Verify all skip connections are used
        for res_idx in range(self.num_resolutions):
            assert len(saved_list[res_idx]) == 0
            assert len(lengths_list[res_idx]) == 0
        assert len(lengths_t) == 0
        assert len(saved_t) == 0

        S = len(self.sources)
        
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
            zout = self._mask(z_list[res_idx], x_list[res_idx])
            zout_list.append(zout)
        
        # Convert back to time domain for each resolution
        if self.use_train_segment:
            if self.training:
                target_length = length
            else:
                target_length = training_length
        else:
            target_length = length
            
        # iSTFT for each resolution with corresponding hop_length
        x_time_list = []
        for res_idx in range(self.num_resolutions):
            x_time = self._ispec(zout_list[res_idx], target_length, hop_length=self.hop_lengths[res_idx])
            x_time_list.append(x_time)

        # Back to MPS device
        if x_is_mps:
            x_time_list = [x.to("mps") for x in x_time_list]

        # Time domain branch processing
        if self.use_train_segment:
            if self.training:
                xt = xt.view(B, S, -1, length)
            else:
                xt = xt.view(B, S, -1, training_length)
        else:
            xt = xt.view(B, S, -1, length)
        xt = xt * stdt[:, None] + meant[:, None]
        
        # Two-stage normalization: column first, then row (no iteration)
        final_weights = torch.exp(self.final_fusion_weights)  # [S, num_res], ensure positive
        # Stage 1: Normalize columns (each resolution/column sums to 1) - dim=0 is sources
        final_weights = final_weights / final_weights.sum(dim=0, keepdim=True)
        # Stage 2: Normalize rows (each source/row sums to 1) - dim=1 is resolutions
        final_weights = final_weights / final_weights.sum(dim=1, keepdim=True)
        
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
        
        # Add time domain branch
        x = xt + x
        
        if length_pre_pad:
            x = x[..., :length_pre_pad]
        return x
