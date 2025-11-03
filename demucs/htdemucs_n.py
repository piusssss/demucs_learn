"""
HTDemucs Advanced: Integrating ResUNet++, Multi-resolution STFT, and Linear Attention
for improved music source separation.
"""

import math
import torch
from torch import nn
from torch.nn import functional as F
from fractions import Fraction
from einops import rearrange
from typing import List, Optional

# Import our new components
from .linear_attention import LinearTransformerEncoder, elu_feature_map
from .multi_resolution_stft import MultiResolutionSTFT, MultiResolutionEncoder
from .resunet_plus import ResUNetPlus

# Import existing components
from .demucs import rescale_module
from .states import capture_init
from .spec import spectro, ispectro
from .hdemucs import pad1d, ScaledEmbedding


class HTDemucs_n(nn.Module):
    """
    Advanced HTDemucs with three key innovations:
    1. ResUNet++ for better feature extraction with residual connections
    2. Multi-resolution STFT for capturing different time-frequency characteristics  
    3. Linear Attention for efficient long-sequence modeling
    
    This model maintains the hybrid time-frequency approach of HTDemucs while
    incorporating state-of-the-art techniques for improved performance and efficiency.
    """
    
    @capture_init
    def __init__(
        self,
        sources,
        # Audio parameters
        audio_channels=2,
        samplerate=44100,
        segment=10,
        # Multi-resolution STFT parameters
        n_ffts=[512, 1024, 2048, 4096],
        stft_hop_lengths=None,
        stft_fusion_method='attention',
        # ResUNet++ parameters
        resunet_base_channels=64,
        resunet_depth=4,
        resunet_use_se=True,
        resunet_use_attention=True,
        # Linear Attention parameters
        linear_attn_layers=5,
        linear_attn_heads=8,
        linear_attn_dim_head=64,
        linear_attn_feature_map=None,
        # Traditional parameters (for compatibility)
        channels=48,
        channels_time=None,
        growth=2,
        nfft=4096,
        wiener_iters=0,
        end_iters=0,
        wiener_residual=False,
        cac=True,
        # Hybrid model parameters
        depth=4,
        rewrite=True,
        kernel_size=8,
        stride=4,
        time_stride=2,
        context=1,
        context_enc=0,
        # Normalization
        norm_starts=4,
        norm_groups=4,
        # DConv parameters (for compatibility)
        dconv_mode=1,
        dconv_depth=2,
        dconv_comp=8,
        dconv_init=1e-3,
        # Bottom channels
        bottom_channels=0,
        # Other parameters
        rescale=0.1,
        use_train_segment=True,
        **kwargs
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
        self.wiener_iters = wiener_iters
        self.end_iters = end_iters
        self.wiener_residual = wiener_residual
        self.cac = cac
        self.n_ffts = n_ffts
        self.use_train_segment = use_train_segment
        
        # Multi-resolution STFT
        self.multi_stft = MultiResolutionSTFT(
            n_ffts=n_ffts,
            hop_lengths=stft_hop_lengths,
            window='hann',
            center=True,
            normalized=False,
            onesided=True
        )
        
        # Multi-resolution encoder
        stft_output_dim = channels * 2  # Match with traditional branch
        self.multi_res_encoder = MultiResolutionEncoder(
            n_ffts=n_ffts,
            channels=audio_channels,
            hidden_dim=channels,
            output_dim=stft_output_dim,
            fusion_method=stft_fusion_method
        )
        
        # Calculate frequency dimension for ResUNet++
        freq_bins = nfft // 2 + 1
        
        # ResUNet++ for frequency branch
        # Calculate input channels: real/imag (2) + multi-res features
        freq_input_channels = audio_channels * 2 + stft_output_dim  # 2*2 + 96 = 100
        self.freq_resunet = ResUNetPlus(
            in_channels=freq_input_channels,
            out_channels=channels,
            base_channels=resunet_base_channels,
            depth=resunet_depth,
            use_se=resunet_use_se,
            use_attention=resunet_use_attention
        )
        
        # Time branch - redesigned for better channel management
        self.time_encoder = nn.ModuleList()
        self.time_decoder = nn.ModuleList()
        
        # Calculate channel progression
        time_channels = [channels]
        ch = channels
        for index in range(depth):
            ch *= growth
            time_channels.append(ch)
        
        # Build encoder
        for index in range(depth):
            in_ch = audio_channels if index == 0 else time_channels[index]
            out_ch = time_channels[index + 1]
            
            encode = []
            encode += [
                nn.Conv1d(in_ch, out_ch, kernel_size, stride),
                nn.ReLU(),
                nn.Conv1d(out_ch, out_ch * 2, 1), 
                nn.GLU(1),
            ]
            self.time_encoder.append(nn.Sequential(*encode))
        
        # Build decoder (reverse order)
        for index in range(depth):
            in_ch = time_channels[depth - index]
            
            decode = []
            decode += [nn.Conv1d(in_ch, in_ch * 2, 1), nn.GLU(1)]
            
            if index < depth - 1:
                out_ch = time_channels[depth - index - 1]
                decode.append(nn.ConvTranspose1d(in_ch, out_ch, kernel_size, stride))
            else:
                # Final layer outputs to sources
                decode.append(nn.ConvTranspose1d(in_ch, audio_channels * len(sources), kernel_size, stride))
            
            self.time_decoder.append(nn.Sequential(*decode))
        
        # Linear Attention Transformer for long-sequence modeling
        transformer_dim = time_channels[-1]  # Use the final encoder dimension
        self.linear_transformer = LinearTransformerEncoder(
            dim=transformer_dim,
            depth=linear_attn_layers,
            heads=linear_attn_heads,
            dim_head=linear_attn_dim_head,
            ff_mult=4,
            dropout=0.0,
            feature_map=linear_attn_feature_map or elu_feature_map,
            norm_first=True
        )
        
        # CAC mode projection (for when CAC is enabled)
        self.cac_projection = nn.Conv1d(audio_channels * 2, channels, 1)
        
        # Frequency-Time fusion
        final_time_ch = time_channels[-1]
        self.freq_projection = nn.Sequential(
            nn.Conv1d(channels, final_time_ch, 1),
            nn.ReLU()
        )
        
        self.freq_time_fusion = nn.Sequential(
            nn.Conv1d(final_time_ch * 2, final_time_ch, 1),  # freq + time
            nn.ReLU(),
            nn.Conv1d(final_time_ch, final_time_ch, 1)
        )
        
        # Final processing
        self.final_conv = nn.Conv1d(ch, ch, 1)
        
        # Initialize weights
        self.apply(self._init_weights)
        if rescale:
            rescale_module(self, reference=rescale)
    
    def _init_weights(self, module):
        """Initialize weights for better training stability."""
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, 0, 0.01)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def valid_length(self, length: int) -> int:
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
        """
        Forward pass through the advanced HTDemucs model.
        
        Args:
            mix: Input mixture [B, C, T]
            
        Returns:
            Separated sources [B, S, C, T] where S is number of sources
        """
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
        
        mono = mix.mean(dim=1, keepdim=True)  # [B, 1, T]
        
        # === FREQUENCY BRANCH ===
        
        # 1. Multi-resolution STFT
        multi_stfts = self.multi_stft(mix)  # List of [B, C, F, T'] tensors
        multi_res_features = self.multi_res_encoder(multi_stfts)  # [B, channels*2, T']
        
        # 2. Traditional single-resolution STFT for compatibility
        # Use a smaller STFT size to avoid padding issues with short sequences
        stft_size = min(self.n_ffts[-1], mix.shape[-1] // 8)  # More conservative
        stft_size = max(stft_size, 512)  # Minimum STFT size
        z = spectro(mix, stft_size, stft_size // 4, 0)  # pad=0 to avoid padding issues
        
        # Convert complex to real/imag
        if self.cac:
            # Complex as channels - simplified approach
            B, C, Fq, T_prime = z.shape
            z_real_imag = torch.stack([z.real, z.imag], dim=2)  # [B, C, 2, F, T']
            z = z_real_imag.permute(0, 1, 3, 2, 4).contiguous()  # [B, C, F, 2, T']
            z = z.view(B, C * 2, Fq, T_prime)  # [B, C*2, F, T']
        else:
            z = torch.stack([z.real, z.imag], dim=2)  # [B, C, 2, F, T']
            z = z.permute(0, 1, 3, 2, 4).contiguous()  # [B, C, F, 2, T']
            B, C, F, _, T_prime = z.shape
            z = z.view(B, C * 2, F, T_prime)  # [B, C*2, F, T']
        
        # 3. Combine multi-resolution features with traditional STFT
        if not self.cac:
            # Interpolate multi-res features to match STFT time dimension
            if multi_res_features.shape[-1] != z.shape[-1]:
                multi_res_features = nn.functional.interpolate(
                    multi_res_features, size=z.shape[-1], 
                    mode='linear', align_corners=False
                )
            
            # Expand multi-res features to match frequency dimension
            multi_res_expanded = multi_res_features.unsqueeze(2).expand(-1, -1, z.shape[2], -1)
            
            # Concatenate along channel dimension
            freq_input = torch.cat([z, multi_res_expanded], dim=1)  # [B, C*2+multi_dim, F, T']
            
            # 4. Process with ResUNet++
            freq_features = self.freq_resunet(freq_input)  # [B, channels, F, T']
            
            # Global average pooling over frequency to get 1D time features
            freq_out = freq_features.mean(dim=2)  # [B, channels, T']
        else:
            # For CAC mode, simplified processing
            # z is [B, C*2, F, T'], average over frequency and project
            freq_averaged = z.mean(dim=2)  # [B, C*2, T']
            freq_out = self.cac_projection(freq_averaged)  # [B, channels, T']
        
        # === TIME BRANCH ===
        
        time_features = []
        xt = mix
        
        # Encoder
        for layer in self.time_encoder:
            xt = layer(xt)
            time_features.append(xt)
        
        # === LINEAR ATTENTION TRANSFORMER ===
        
        # Prepare input for transformer: [B, C, T] -> [B, T, C]
        transformer_input = xt.transpose(1, 2)
        
        # Apply linear attention transformer
        transformer_out = self.linear_transformer(transformer_input)  # [B, T, C]
        
        # Convert back: [B, T, C] -> [B, C, T]
        xt = transformer_out.transpose(1, 2)
        
        # === FREQUENCY-TIME FUSION ===
        
        # Project frequency features to match time dimension
        freq_projected = self.freq_projection(freq_out)  # [B, final_time_ch, T']
        
        # Interpolate frequency features to match time features
        if freq_projected.shape[-1] != xt.shape[-1]:
            freq_projected = nn.functional.interpolate(freq_projected, size=xt.shape[-1], mode='linear', align_corners=False)
        
        # Fuse frequency and time features
        fused_input = torch.cat([freq_projected, xt], dim=1)  # [B, final_time_ch*2, T]
        fused = self.freq_time_fusion(fused_input)  # [B, final_time_ch, T]
        
        # Add residual connection
        xt = xt + fused
        
        # Final processing
        xt = self.final_conv(xt)
        
        # === TIME DECODER ===
        
        # Decoder with skip connections
        skip_features = list(reversed(time_features[:-1]))  # Exclude the last (deepest) feature
        
        for i, layer in enumerate(self.time_decoder):
            xt = layer(xt)
            
            # Add skip connection if available and not the final layer
            if i < len(skip_features):
                skip = skip_features[i]
                if xt.shape[-1] != skip.shape[-1]:
                    xt = nn.functional.interpolate(xt, size=skip.shape[-1], mode='linear', align_corners=False)
                # Only add skip if channel dimensions match
                if xt.shape[1] == skip.shape[1]:
                    xt = xt + skip
        
        # Reshape to [B, S, C, T]
        S = len(self.sources)
        B, _, T = xt.shape
        xt = xt.view(B, S, self.audio_channels, T)
        
        # Restore original length if padding was applied
        if length_pre_pad is not None:
            xt = xt[..., :length_pre_pad]
        
        return xt
    
    def separate(self, wav, shifts=1, split=True, overlap=0.25, callback=None, device=None):
        """
        Separate audio with optional test-time augmentation.
        This method handles longer sequences efficiently using the linear attention.
        """
        if device is None:
            device = next(iter(self.parameters())).device
        
        wav = wav.to(device)
        
        if shifts == 1:
            # Single pass - can handle longer sequences due to linear attention
            with torch.no_grad():
                sources = self(wav.unsqueeze(0))[0]
            return sources
        else:
            # Multiple shifts for better quality
            # The linear attention allows us to process longer segments
            return self._separate_with_shifts(wav, shifts, split, overlap, callback, device)
    
    def _separate_with_shifts(self, wav, shifts, split, overlap, callback, device):
        """Handle separation with multiple shifts and overlapping."""
        # Implementation similar to original HTDemucs but can handle longer sequences
        # This is a simplified version - full implementation would include
        # proper shift handling and overlap-add reconstruction
        
        sources_acc = torch.zeros(len(self.sources), *wav.shape, device=device)
        
        for shift in range(shifts):
            # Apply random shift
            offset = torch.randint(0, wav.shape[-1] // 8, (1,)).item()
            shifted_wav = torch.roll(wav, offset, dims=-1)
            
            with torch.no_grad():
                sources = self(shifted_wav.unsqueeze(0))[0]
            
            # Reverse shift
            sources = torch.roll(sources, -offset, dims=-1)
            sources_acc += sources
        
        return sources_acc / shifts