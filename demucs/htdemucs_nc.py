import math

import torch
from torch import nn
from torch.nn import functional as F
from fractions import Fraction

from .demucs import rescale_module
from .states import capture_init
from .hdemucs import pad1d, HEncLayer, HDecLayer


class HTDemucs_nc(nn.Module):

    @capture_init
    def __init__(
        self,
        sources,
        # Channels
        audio_channels=2,
        channels=24,
        growth=2,
        # Main structure
        depth=7,
        rewrite=False,
        # Convolutions
        kernel_size=8,
        stride=2,
        context=1,
        context_enc=0,
        # Normalization
        norm_starts=4,
        norm_groups=4,
        # DConv residual branch
        dconv_mode=3,
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

        self.tencoder = nn.ModuleList()
        self.tdecoder = nn.ModuleList()
        
        chin = audio_channels
        chout = channels

        for index in range(depth):
            norm = index >= norm_starts
            stri = stride
            ker = kernel_size
            freq = False
            pad = True

            kwt = {
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
            kw_dec = dict(kwt)

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
            chout = int(growth * chout)

        if rescale:
            rescale_module(self, reference=rescale)

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

        # Prepare the time branch input.
        xt = mix
        meant = xt.mean(dim=(1, 2), keepdim=True)
        stdt = xt.std(dim=(1, 2), keepdim=True)
        xt = (xt - meant) / (1e-5 + stdt)
        B, C, T = xt.shape
        
        saved_t = []  # skip connections, time branch (shared)
        lengths_t = []  # saved lengths for time branch (shared)
        for idx in range(self.depth):
            #print(f"Debug - {idx}  , y.shape: {xt.shape}")
            
            lengths_t.append(xt.shape[-1])
            tenc = self.tencoder[idx]
            xt = tenc(xt)
            saved_t.append(xt)

        for idx in range(self.depth):
            tdec = self.tdecoder[idx]
            length_t = lengths_t.pop(-1)
            skip = saved_t.pop(-1)
            xt, _ = tdec(xt, skip, length_t)
            
            #print(f"Debug - {idx}  , y.shape: {xt.shape}")
        # Let's make sure we used all stored skip connections.
        assert len(lengths_t) == 0
        assert len(saved_t) == 0

        S = len(self.sources)

        if self.use_train_segment:
            if self.training:
                xt = xt.view(B, S, -1, length)
            else:
                xt = xt.view(B, S, -1, training_length)
        else:
            xt = xt.view(B, S, -1, length)
        xt = xt * stdt[:, None] + meant[:, None]
        if length_pre_pad:
            xt = xt[..., :length_pre_pad]
        return xt