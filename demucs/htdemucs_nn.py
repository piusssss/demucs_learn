import math

from openunmix.filtering import wiener
import torch
from torch import nn
from torch.nn import functional as F
from fractions import Fraction
from einops import rearrange

from .transformer_n import CrossTransformerEncoder

from .demucs import rescale_module
from .states import capture_init
from .spec import spectro, ispectro
from .hdemucs import pad1d, ScaledEmbedding, HEncLayer, HDecLayer


class HTDemucs_nn(nn.Module):

    @capture_init
    def __init__(
        self,
        sources,
        # Channels
        audio_channels=2,
        channels=24,
        channels_time=None,
        growth=2,
        # STFT
        wiener_iters=0,
        end_iters=0,
        wiener_residual=False,
        cac=True,
        # Main structure
        depth=4,
        rewrite=True,
        # Frequency branch
        freq_emb=0.3,
        emb_scale=10,
        emb_smooth=True,
        # Convolutions
        kernel_size=8,
        time_stride=2,
        stride=4,
        context=1,
        context_enc=0,
        # Normalization
        norm_starts=3,
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
        # Multi-resolution window sizes
        self.window_sizes = [2048, 4096, 8192]
        self.hop_lengths = [ws // 4 for ws in self.window_sizes]
        self.wiener_iters = wiener_iters
        self.end_iters = end_iters
        self.freq_emb = None
        assert wiener_iters == end_iters

        # Multi-resolution encoders
        self.encoder_2048 = nn.ModuleList()
        self.encoder_4096 = nn.ModuleList()
        self.encoder_8192 = nn.ModuleList()
        
        # Multi-resolution decoders
        self.decoder_2048 = nn.ModuleList()
        self.decoder_4096 = nn.ModuleList()
        self.decoder_8192 = nn.ModuleList()

        self.tencoder = nn.ModuleList()
        self.tdecoder = nn.ModuleList()
        
        # Multi-resolution fusion using convolution (no learnable weights)
        # Use transposed convolution to split fused features back to 3 resolutions
        
        # Get transformer output channels
        transformer_channels = channels * growth ** (depth - 1)
        if bottom_channels:
            fusion_channels = bottom_channels
        else:
            fusion_channels = transformer_channels
        
        # Pre-transformer: 3 resolutions → 1 fused (Conv2d)
        # Calculate frequency dimension after all encoder layers
        # For 4096 resolution: freq = (4096 // 2) // (stride ** depth)
        nfft_target = 4096
        freq_after_encoders = (nfft_target // 2) // (stride ** depth)
        
        # Total channels after reshaping C and F dimensions
        total_fusion_channels = fusion_channels * freq_after_encoders
        channels_per_group = freq_after_encoders
        
        # Input: [B, C*F, 3, T] → Output: [B, C*F, 1, T]
        self.pre_transformer_fusion = nn.Conv2d(
            in_channels=total_fusion_channels,
            out_channels=total_fusion_channels,
            kernel_size=[3, 129],
            stride=1,
            padding=[0, 64],
            groups=total_fusion_channels // channels_per_group  # Similar to final fusion
        )
        
        # Post-transformer: 1 fused → 3 resolutions (ConvTranspose2d)
        # Input: [B, C*F, 1, T] → Output: [B, C*F, 3, T]
        self.post_transformer_split = nn.ConvTranspose2d(
            in_channels=total_fusion_channels,
            out_channels=total_fusion_channels,
            kernel_size=[3, 129],
            stride=1,
            padding=[0, 64],
            groups=total_fusion_channels // channels_per_group  # Similar to final fusion
        )
        
        # Final time-domain fusion (keep this as is)
        num_groups = len(self.sources) * self.audio_channels
        fusion_conv_wide = 129
        self.fusion_conv = nn.Conv2d(
            in_channels=num_groups, 
            out_channels=num_groups,  
            kernel_size=[3, fusion_conv_wide],  
            stride=1,
            padding=[0, (fusion_conv_wide-1)//2],
            groups=num_groups//self.audio_channels  
        )
        
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

            # Create three resolution encoders
            enc_2048 = HEncLayer(
                chin_z, chout_z, dconv=dconv_mode & 1, context=context_enc, **kw
            )
            enc_4096 = HEncLayer(
                chin_z, chout_z, dconv=dconv_mode & 1, context=context_enc, **kw
            )
            enc_8192 = HEncLayer(
                chin_z, chout_z, dconv=dconv_mode & 1, context=context_enc, **kw
            )
            if freq:
                tenc = HEncLayer(
                    chin,
                    chout,
                    dconv=dconv_mode & 1,
                    context=context_enc,
                    **kwt
                )
                self.tencoder.append(tenc)
            
            self.encoder_2048.append(enc_2048)
            self.encoder_4096.append(enc_4096)
            self.encoder_8192.append(enc_8192)
            if index == 0:
                chin = self.audio_channels * len(self.sources)
                chin_z = chin
                if self.cac:
                    chin_z *= 2
            # Create three resolution decoders
            dec_2048 = HDecLayer(
                chout_z,
                chin_z,
                dconv=dconv_mode & 2,
                last=index == 0,
                context=context,
                **kw_dec
            )
            dec_4096 = HDecLayer(
                chout_z,
                chin_z,
                dconv=dconv_mode & 2,
                last=index == 0,
                context=context,
                **kw_dec
            )
            dec_8192 = HDecLayer(
                chout_z,
                chin_z,
                dconv=dconv_mode & 2,
                last=index == 0,
                context=context,
                **kw_dec
            )
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
            self.decoder_2048.insert(0, dec_2048)
            self.decoder_4096.insert(0, dec_4096)
            self.decoder_8192.insert(0, dec_8192)

            chin = chout
            chin_z = chout_z
            chout = int(growth * chout)
            chout_z = int(growth * chout_z)

            if index == 0 and freq_emb:
                # Create frequency embeddings for three resolutions
                # After first layer: freqs = (nfft // 2) // stride
                freq_after_first_layer_2048 = (2048 // 2) // stride
                freq_after_first_layer_4096 = (4096 // 2) // stride  
                freq_after_first_layer_8192 = (8192 // 2) // stride
                
                self.freq_emb_2048 = ScaledEmbedding(
                    freq_after_first_layer_2048, chin_z, smooth=emb_smooth, scale=emb_scale
                )
                self.freq_emb_4096 = ScaledEmbedding(
                    freq_after_first_layer_4096, chin_z, smooth=emb_smooth, scale=emb_scale
                )
                self.freq_emb_8192 = ScaledEmbedding(
                    freq_after_first_layer_8192, chin_z, smooth=emb_smooth, scale=emb_scale
                )
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
        assert hl == nfft // 4

        # We re-pad the signal in order to keep the property
        # that the size of the output is exactly the size of the input
        # divided by the stride (here hop_length), when divisible.
        # This is achieved by padding by 1/4th of the kernel size (here nfft).
        # which is not supported by torch.stft.
        # Having all convolution operations follow this convention allow to easily
        # align the time and frequency branches later on.
        assert hl == nfft // 4
        le = int(math.ceil(x.shape[-1] / hl))
        pad = hl // 2 * 3
        x = pad1d(x, (pad, pad + le * hl - x.shape[-1]), mode="reflect")

        z = spectro(x, nfft, hl)[..., :-1, :]
        assert z.shape[-1] == le + 4, (z.shape, x.shape, le)
        z = z[..., 2: 2 + le]
        return z

    def _ispec(self, z, length=None, scale=0, hop_length=None):
        hl = hop_length // (4**scale) 
        z = F.pad(z, (0, 0, 0, 1))
        z = F.pad(z, (2, 2))
        pad = hl // 2 * 3
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
        # Multi-resolution STFT
        z_2048 = self._spec(mix, nfft=2048, hop_length=self.hop_lengths[0])
        z_4096 = self._spec(mix, nfft=4096, hop_length=self.hop_lengths[1])
        z_8192 = self._spec(mix, nfft=8192, hop_length=self.hop_lengths[2])
        
        # Convert to magnitude
        mag_2048 = self._magnitude(z_2048).to(mix.device)
        mag_4096 = self._magnitude(z_4096).to(mix.device)
        mag_8192 = self._magnitude(z_8192).to(mix.device)
        
        # Get shapes for each resolution
        B, C, Fq_2048, T_2048 = mag_2048.shape
        _, _, Fq_4096, T_4096 = mag_4096.shape
        _, _, Fq_8192, T_8192 = mag_8192.shape

        # Normalize each resolution separately
        # 2048 resolution
        mean_2048 = mag_2048.mean(dim=(1, 2, 3), keepdim=True)
        std_2048 = mag_2048.std(dim=(1, 2, 3), keepdim=True)
        x_2048 = (mag_2048 - mean_2048) / (1e-5 + std_2048)
        
        # 4096 resolution  
        mean_4096 = mag_4096.mean(dim=(1, 2, 3), keepdim=True)
        std_4096 = mag_4096.std(dim=(1, 2, 3), keepdim=True)
        x_4096 = (mag_4096 - mean_4096) / (1e-5 + std_4096)
        
        # 8192 resolution
        mean_8192 = mag_8192.mean(dim=(1, 2, 3), keepdim=True)
        std_8192 = mag_8192.std(dim=(1, 2, 3), keepdim=True)
        x_8192 = (mag_8192 - mean_8192) / (1e-5 + std_8192)

        # Prepare the time branch input.
        xt = mix
        meant = xt.mean(dim=(1, 2), keepdim=True)
        stdt = xt.std(dim=(1, 2), keepdim=True)
        xt = (xt - meant) / (1e-5 + stdt)

        # Multi-resolution skip connections and lengths
        saved_2048 = []  # skip connections for 2048 resolution
        saved_4096 = []  # skip connections for 4096 resolution  
        saved_8192 = []  # skip connections for 8192 resolution
        saved_t = []  # skip connections, time branch (shared)
        
        lengths_2048 = []  # saved lengths for 2048 resolution
        lengths_4096 = []  # saved lengths for 4096 resolution
        lengths_8192 = []  # saved lengths for 8192 resolution
        lengths_t = []  # saved lengths for time branch (shared)
        for idx in range(self.depth):
            #print(f"Debug - {idx}  {x_2048.shape} {x_4096.shape} {x_8192.shape}, y.shape: {xt.shape}")
            # Save lengths for each resolution
            lengths_2048.append(x_2048.shape[-1])
            lengths_4096.append(x_4096.shape[-1])
            lengths_8192.append(x_8192.shape[-1])
            inject = None
            lengths_t.append(xt.shape[-1])
            tenc = self.tencoder[idx]
            xt = tenc(xt)
            saved_t.append(xt)
            # Encode three resolutions in parallel
            x_2048 = self.encoder_2048[idx](x_2048, inject)
            x_4096 = self.encoder_4096[idx](x_4096, inject)
            x_8192 = self.encoder_8192[idx](x_8192, inject)
            if idx == 0 and self.freq_emb_scale is not None:
                # Add frequency embedding for each resolution
                # 2048 resolution
                frs_2048 = torch.arange(x_2048.shape[-2], device=x_2048.device)
                emb_2048 = self.freq_emb_2048(frs_2048).t()[None, :, :, None].expand_as(x_2048)
                x_2048 = x_2048 + self.freq_emb_scale * emb_2048
                
                # 4096 resolution
                frs_4096 = torch.arange(x_4096.shape[-2], device=x_4096.device)
                emb_4096 = self.freq_emb_4096(frs_4096).t()[None, :, :, None].expand_as(x_4096)
                x_4096 = x_4096 + self.freq_emb_scale * emb_4096
                
                # 8192 resolution
                frs_8192 = torch.arange(x_8192.shape[-2], device=x_8192.device)
                emb_8192 = self.freq_emb_8192(frs_8192).t()[None, :, :, None].expand_as(x_8192)
                x_8192 = x_8192 + self.freq_emb_scale * emb_8192

            # Save skip connections for each resolution
            saved_2048.append(x_2048)
            saved_4096.append(x_4096)
            saved_8192.append(x_8192)
                   
        if self.crosstransformer:
            # Save pre-fusion states for residual connections
            pre_fusion_2048 = x_2048.clone()
            pre_fusion_4096 = x_4096.clone()
            pre_fusion_8192 = x_8192.clone()
            
            # Get shapes for each resolution
            B_2048, C_2048, F_2048, T_2048 = x_2048.shape
            B_4096, C_4096, F_4096, T_4096 = x_4096.shape
            B_8192, C_8192, F_8192, T_8192 = x_8192.shape
            
            # Use 4096 as target resolution (middle resolution)
            target_F, target_T = F_4096, T_4096
            B, C = B_4096, C_4096
            
            # Interpolate 2048 and 8192 to match 4096's shape
            x_2048_aligned = F.interpolate(x_2048, size=(target_F, target_T), mode='bilinear', align_corners=False)
            x_8192_aligned = F.interpolate(x_8192, size=(target_F, target_T), mode='bilinear', align_corners=False)
            
            # Now all three have the same shape: [B, C, target_F, target_T]
            # Stack (same as final fusion): [B, C, 3, target_F, target_T]
            x_stacked = torch.stack([x_2048_aligned, x_4096, x_8192_aligned], dim=2)
            
            # Permute to: [B, C, target_F, 3, target_T]
            x_stacked = x_stacked.permute(0, 1, 3, 2, 4)
            
            # Reshape for convolution: [B, C*target_F, 3, target_T]
            x_stacked = x_stacked.reshape(B, C * target_F, 3, target_T)
            
            # Fuse 3 resolutions into 1: [B, C*target_F, 3, target_T] → [B, C*target_F, 1, target_T]
            x_fused = self.pre_transformer_fusion(x_stacked)  # [B, C*target_F, 1, target_T]
            x_fused = x_fused.squeeze(2)  # [B, C*target_F, target_T]
            
            # Reshape back: [B, C, target_F, target_T]
            x = x_fused.reshape(B, C, target_F, target_T)
            
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
            
            # Split fused feature back to 3 resolutions using transposed convolution
            # Reshape: [B, C, target_F, target_T] → [B, C*target_F, 1, target_T]
            B, C, target_F, target_T = x.shape
            x_reshaped = x.reshape(B, C * target_F, 1, target_T)
            
            # Split: [B, C*target_F, 1, target_T] → [B, C*target_F, 3, target_T]
            x_split = self.post_transformer_split(x_reshaped)  # [B, C*target_F, 3, target_T]
            
            # Reshape back: [B, C*target_F, 3, target_T] → [B, C, target_F, 3, target_T]
            x_split = x_split.reshape(B, C, target_F, 3, target_T)
            
            # Permute to: [B, C, 3, target_F, target_T] (inverse of pre-transformer)
            x_split = x_split.permute(0, 1, 3, 2, 4)
            
            # Extract three resolutions: each is [B, C, target_F, target_T]
            x_2048_split = x_split[:, :, 0, :, :]
            x_4096_split = x_split[:, :, 1, :, :]
            x_8192_split = x_split[:, :, 2, :, :]
            
            # Interpolate back to original shapes
            x_2048_split = F.interpolate(x_2048_split, size=(F_2048, T_2048), mode='bilinear', align_corners=False)
            x_4096_split = F.interpolate(x_4096_split, size=(F_4096, T_4096), mode='bilinear', align_corners=False)
            x_8192_split = F.interpolate(x_8192_split, size=(F_8192, T_8192), mode='bilinear', align_corners=False)
            
            # Add residual connections
            x_2048 = x_2048_split + pre_fusion_2048
            x_4096 = x_4096_split + pre_fusion_4096
            x_8192 = x_8192_split + pre_fusion_8192

        for idx in range(self.depth):
            # Pop skip connections for each resolution
            skip_2048 = saved_2048.pop(-1)
            skip_4096 = saved_4096.pop(-1)
            skip_8192 = saved_8192.pop(-1)

            # Decode each resolution separately
            x_2048, _ = self.decoder_2048[idx](x_2048, skip_2048, lengths_2048.pop(-1))
            x_4096, _ = self.decoder_4096[idx](x_4096, skip_4096, lengths_4096.pop(-1))
            x_8192, _ = self.decoder_8192[idx](x_8192, skip_8192, lengths_8192.pop(-1))
            # `pre` contains the output just before final transposed convolution,
            # which is used when the freq. and time branch separate.

            # Time domain decoder (shared across resolutions)
            tdec = self.tdecoder[idx]
            length_t = lengths_t.pop(-1)
            skip = saved_t.pop(-1)
            xt, _ = tdec(xt, skip, length_t)
            #print(f"Debug - {idx}  {x_2048.shape} {x_4096.shape} {x_8192.shape}, y.shape: {xt.shape}")
        # Let's make sure we used all stored skip connections.
        assert len(saved_2048) == 0
        assert len(saved_4096) == 0
        assert len(saved_8192) == 0
        assert len(lengths_2048) == 0
        assert len(lengths_4096) == 0
        assert len(lengths_8192) == 0
        assert len(lengths_t) == 0
        assert len(saved_t) == 0

        S = len(self.sources)
        
        # Process each resolution separately
        # 2048 resolution
        x_2048 = x_2048.view(B, S, -1, Fq_2048, T_2048)
        x_2048 = x_2048 * std_2048[:, None] + mean_2048[:, None]
        
        # 4096 resolution  
        x_4096 = x_4096.view(B, S, -1, Fq_4096, T_4096)
        x_4096 = x_4096 * std_4096[:, None] + mean_4096[:, None]
        
        # 8192 resolution
        x_8192 = x_8192.view(B, S, -1, Fq_8192, T_8192)
        x_8192 = x_8192 * std_8192[:, None] + mean_8192[:, None]

        # to cpu as mps doesnt support complex numbers
        # demucs issue #435 ##432
        # NOTE: in this case z already is on cpu
        # TODO: remove this when mps supports complex numbers
        x_is_mps = x_4096.device.type == "mps"
        if x_is_mps:
            x_2048 = x_2048.cpu()
            x_4096 = x_4096.cpu()
            x_8192 = x_8192.cpu()

        # Apply masking for each resolution
        zout_2048 = self._mask(z_2048, x_2048)
        zout_4096 = self._mask(z_4096, x_4096)
        zout_8192 = self._mask(z_8192, x_8192)
        
        # Convert back to time domain for each resolution
        if self.use_train_segment:
            if self.training:
                target_length = length
            else:
                target_length = training_length
        else:
            target_length = length
            
        # iSTFT for each resolution with corresponding hop_length
        x_2048 = self._ispec(zout_2048, target_length, hop_length=self.hop_lengths[0])
        x_4096 = self._ispec(zout_4096, target_length, hop_length=self.hop_lengths[1])
        x_8192 = self._ispec(zout_8192, target_length, hop_length=self.hop_lengths[2])

        # back to mps device
        if x_is_mps:
            x_2048 = x_2048.to("mps")
            x_4096 = x_4096.to("mps")
            x_8192 = x_8192.to("mps")

        # Time domain branch processing
        if self.use_train_segment:
            if self.training:
                xt = xt.view(B, S, -1, length)
            else:
                xt = xt.view(B, S, -1, training_length)
        else:
            xt = xt.view(B, S, -1, length)
        xt = xt * stdt[:, None] + meant[:, None]
        
        # Multi-resolution time domain fusion using convolution
        # Stack the three resolutions
        x_time_list = [x_2048, x_4096, x_8192]
        x_stacked = torch.stack(x_time_list, dim=2)  # [B, S, 3, C, T]
        B, S, num_res, C, T = x_stacked.shape
        x_stacked = x_stacked.permute(0, 1, 3, 2, 4)  # [B, S, C, 3, T]
        x_stacked = x_stacked.reshape(B, S * C, num_res, T)  # [B, S*C, 3, T]
        
        x = self.fusion_conv(x_stacked)  # [B, S*C, 1, T]
        x = x.squeeze(2)  # [B, S*C, T]
        x = x.reshape(B, S, C, T)  # [B, S, C, T]
        
        # Add time domain branch
        x = xt + x
        
        if length_pre_pad:
            x = x[..., :length_pre_pad]
        return x
