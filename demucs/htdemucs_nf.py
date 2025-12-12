import math

import torch
from torch import nn
from torch.nn import functional as F
from fractions import Fraction

from .demucs import rescale_module
from .states import capture_init
from .spec import spectro, ispectro
from .hdemucs import pad1d, ScaledEmbedding, HEncLayer, HDecLayer


class HTDemucs_nf(nn.Module):

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
        nfft_list=[2048, 4096, 8192, 16384],  # Multi-resolution STFT window sizes 
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
        # Transformer
        t_layers=1,
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
        t_cross_first=True,
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
        self.use_train_segment = use_train_segment
        
        # Multi-resolution window sizes
        self.nfft_list = nfft_list
        self.num_resolutions = len(nfft_list)
        self.hop_lengths = [nfft // 2 for nfft in nfft_list]

        # Multi-resolution encoders and decoders (dynamic based on nfft_list)
        self.encoders = nn.ModuleList([nn.ModuleList() for _ in range(self.num_resolutions)])
        self.decoders = nn.ModuleList([nn.ModuleList() for _ in range(self.num_resolutions)])
        

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
            kw_dec = dict(kw)

            # Create encoders for each resolution
            for res_idx in range(self.num_resolutions):
                enc = HEncLayer(
                    chin_z, chout_z, dconv=dconv_mode & 1, context=context_enc, **kw
                )
                self.encoders[res_idx].append(enc)
            
            if index == 0:
                chin = self.audio_channels * len(self.sources)
                chin_z = chin
                if self.cac:
                    chin_z *= 2
            
            # Create decoders for each resolution
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

            chin = chout
            chin_z = chout_z
            chout = int(growth * chout)
            chout_z = int(growth * chout_z)

            if index == 0 and freq_emb:
                # Create frequency embeddings for each resolution
                # After first layer: freqs = (nfft // 2) // stride
                self.freq_embeddings = nn.ModuleList()
                for nfft in self.nfft_list:
                    freq_after_first_layer = (nfft // 2) // stride
                    freq_emb_layer = ScaledEmbedding(
                        freq_after_first_layer, chin_z, smooth=emb_smooth, scale=emb_scale
                    )
                    self.freq_embeddings.append(freq_emb_layer)
                self.freq_emb_scale = freq_emb

        if rescale:
            rescale_module(self, reference=rescale)

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
        
        for nfft, hop_length in zip(self.nfft_list, self.hop_lengths):
            z = self._spec(mix, nfft=nfft, hop_length=hop_length)
            mag = self._magnitude(z).to(mix.device)
            
            B, C, Fq, T = mag.shape
            shapes_list.append((Fq, T))
            
            # Normalize
            mean = mag.mean(dim=(1, 2, 3), keepdim=True)
            std = mag.std(dim=(1, 2, 3), keepdim=True)
            x = (mag - mean) / (1e-5 + std)
            
            mean_list.append(mean)
            std_list.append(std)
            x_list.append(x)
        
        # Multi-resolution skip connections and lengths
        saved_list = [[] for _ in range(self.num_resolutions)]
        lengths_list = [[] for _ in range(self.num_resolutions)]
        
        for idx in range(self.depth):
            # Save lengths for each resolution
            for res_idx in range(self.num_resolutions):
                lengths_list[res_idx].append(x_list[res_idx].shape[-1])
            
            inject = None
            
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
            
        # Decode all resolutions in parallel
        for idx in range(self.depth):
            for res_idx in range(self.num_resolutions):
                skip = saved_list[res_idx].pop(-1)
                target_len = lengths_list[res_idx].pop(-1)
                x_list[res_idx], _ = self.decoders[res_idx][idx](x_list[res_idx], skip, target_len)
        
       # Verify all skip connections are used
        for res_idx in range(self.num_resolutions):
            assert len(saved_list[res_idx]) == 0
            assert len(lengths_list[res_idx]) == 0

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
            
        # iSTFT for each resolution with corresponding hop_length
        x_time_list = []
        for res_idx in range(self.num_resolutions):
            x_time = self._ispec(zout_list[res_idx], target_length, hop_length=self.hop_lengths[res_idx])
            x_time_list.append(x_time)

        # Back to MPS device
        if x_is_mps:
            x_time_list = [x.to("mps") for x in x_time_list]

        fixed_assignment = [0, 3, 2, 1]  # drums->2048, bass->16384, other->8192, vocals->4096
        
        B, S, C, T = x_time_list[0].shape
        x = torch.zeros(B, S, C, T, device=x_time_list[0].device, dtype=x_time_list[0].dtype)
        
        for s in range(S):
            x[:, s] = x_time_list[fixed_assignment[s]][:, s] 
        
        if length_pre_pad:
            x = x[..., :length_pre_pad]
        return x