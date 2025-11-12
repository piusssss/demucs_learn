import math

from openunmix.filtering import wiener
import torch
from torch import nn
from torch.nn import functional as F
from fractions import Fraction
from einops import rearrange

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
        channels=48,
        channels_time=None,
        growth=2,
        # STFT
        nfft_list=[1024, 2048, 4096, 8192, 16384],  # Multi-resolution STFT window sizes
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
        stride=4,
        context=1,
        context_enc=0,
        # Normalization
        norm_starts=3,
        norm_groups=4,
        # DConv residual branch
        dconv_mode=0,
        dconv_depth=2,
        dconv_comp=8,
        dconv_init=1e-3,
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
        self.channels = channels
        self.samplerate = samplerate
        self.segment = segment
        self.use_train_segment = use_train_segment
        
        # Multi-resolution window sizes
        self.nfft_list = nfft_list
        self.num_resolutions = len(nfft_list)
        self.hop_lengths = [nfft // 4 for nfft in nfft_list]
        self.wiener_iters = wiener_iters
        self.end_iters = end_iters
        self.freq_emb = None
        assert wiener_iters == end_iters

        # Multi-resolution encoders and decoders (dynamic based on nfft_list)
        self.encoders = nn.ModuleList([nn.ModuleList() for _ in range(self.num_resolutions)])
        self.decoders = nn.ModuleList([nn.ModuleList() for _ in range(self.num_resolutions)])
        
        # Multi-resolution fusion convolution
        num_groups = len(self.sources) * self.audio_channels
        fusion_conv_wide=129
        self.fusion_conv = nn.Conv2d(
            in_channels=num_groups, 
            out_channels=num_groups,  
            kernel_size=[self.num_resolutions, fusion_conv_wide],  
            stride=1,
            padding=[0,(fusion_conv_wide-1)//2],
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
        
        # Multi-resolution fusion using convolution
        x_stacked = torch.stack(x_time_list, dim=2)  # [B, S, 5, C, T]
        B, S, num_res, C, T = x_stacked.shape
        
        x_stacked = x_stacked.permute(0, 1, 3, 2, 4)  # [B, S, C, 5, T]
        
        x_stacked = x_stacked.reshape(B, S * C, num_res, T)  # [B, S*C, 5, T]
        
        x = self.fusion_conv(x_stacked)  # [B, S*C, 1, T]
        
        x = x.squeeze(2)  # [B, S*C, T] 
        x = x.reshape(B, S, C, T)  # [B, S, C, T] 
        if length_pre_pad:
            x = x[..., :length_pre_pad]
        return x
