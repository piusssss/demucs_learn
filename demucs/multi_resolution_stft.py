"""
Multi-resolution STFT implementation for capturing different time-frequency characteristics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import math


class MultiResolutionSTFT(nn.Module):
    """
    Multi-resolution STFT that computes STFT with multiple window sizes.
    
    Args:
        n_ffts: List of FFT sizes for different resolutions
        hop_lengths: List of hop lengths (if None, uses n_fft // 4)
        win_lengths: List of window lengths (if None, uses n_fft)
        window: Window function name
        center: Whether to center the STFT
        normalized: Whether to normalize the STFT
        onesided: Whether to return one-sided STFT
    """
    
    def __init__(
        self,
        n_ffts: List[int] = [512, 1024, 2048, 4096],
        hop_lengths: Optional[List[int]] = None,
        win_lengths: Optional[List[int]] = None,
        window: str = 'hann',
        center: bool = True,
        normalized: bool = False,
        onesided: bool = True
    ):
        super().__init__()
        
        self.n_ffts = n_ffts
        self.hop_lengths = hop_lengths or [n_fft // 4 for n_fft in n_ffts]
        self.win_lengths = win_lengths or n_ffts
        self.window = window
        self.center = center
        self.normalized = normalized
        self.onesided = onesided
        
        # Register window buffers
        for i, (n_fft, win_length) in enumerate(zip(self.n_ffts, self.win_lengths)):
            window_tensor = self._get_window(window, win_length)
            self.register_buffer(f'window_{i}', window_tensor)
    
    def _get_window(self, window: str, win_length: int) -> torch.Tensor:
        """Generate window function."""
        if window == 'hann':
            return torch.hann_window(win_length)
        elif window == 'hamming':
            return torch.hamming_window(win_length)
        elif window == 'blackman':
            return torch.blackman_window(win_length)
        else:
            raise ValueError(f"Unsupported window: {window}")
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Compute multi-resolution STFT.
        
        Args:
            x: Input waveform [B, C, T]
            
        Returns:
            List of STFT tensors [B, C, F, T'] for each resolution
        """
        stfts = []
        
        for i, (n_fft, hop_length, win_length) in enumerate(
            zip(self.n_ffts, self.hop_lengths, self.win_lengths)
        ):
            window = getattr(self, f'window_{i}')
            
            # Compute STFT for current resolution
            stft = torch.stft(
                x.reshape(-1, x.shape[-1]),  # [B*C, T]
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=window,
                center=self.center,
                normalized=self.normalized,
                onesided=self.onesided,
                return_complex=True
            )
            
            # Reshape back to [B, C, F, T']
            B, C = x.shape[:2]
            F, T_prime = stft.shape[-2:]
            stft = stft.reshape(B, C, F, T_prime)
            
            stfts.append(stft)
        
        return stfts
    
    def inverse(self, stfts: List[torch.Tensor], length: Optional[int] = None) -> torch.Tensor:
        """
        Compute inverse STFT. Uses the first (highest resolution) STFT for reconstruction.
        
        Args:
            stfts: List of STFT tensors
            length: Target length for output
            
        Returns:
            Reconstructed waveform [B, C, T]
        """
        # Use the first STFT (typically highest time resolution) for reconstruction
        stft = stfts[0]
        B, C, F, T_prime = stft.shape
        
        # Get corresponding parameters
        n_fft = self.n_ffts[0]
        hop_length = self.hop_lengths[0]
        win_length = self.win_lengths[0]
        window = getattr(self, 'window_0')
        
        # Compute inverse STFT
        x = torch.istft(
            stft.reshape(-1, F, T_prime),  # [B*C, F, T']
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=self.center,
            normalized=self.normalized,
            onesided=self.onesided,
            length=length
        )
        
        # Reshape back to [B, C, T]
        if length is not None:
            T = length
        else:
            T = x.shape[-1]
        
        return x.reshape(B, C, T)


class MultiResolutionEncoder(nn.Module):
    """
    Encoder that processes multi-resolution STFT features.
    
    Args:
        n_ffts: List of FFT sizes
        channels: Number of input channels
        hidden_dim: Hidden dimension for processing each resolution
        output_dim: Output dimension after fusion
        fusion_method: Method to fuse multi-resolution features ('concat', 'add', 'attention')
    """
    
    def __init__(
        self,
        n_ffts: List[int] = [512, 1024, 2048, 4096],
        channels: int = 2,
        hidden_dim: int = 64,
        output_dim: int = 256,
        fusion_method: str = 'attention'
    ):
        super().__init__()
        
        self.n_ffts = n_ffts
        self.fusion_method = fusion_method
        
        # Individual encoders for each resolution
        self.encoders = nn.ModuleList()
        for n_fft in n_ffts:
            freq_bins = n_fft // 2 + 1  # For onesided STFT
            encoder = nn.Sequential(
                nn.Conv2d(channels * 2, hidden_dim, kernel_size=3, padding=1),  # *2 for real/imag
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True)
            )
            self.encoders.append(encoder)
        
        # Fusion layer
        if fusion_method == 'concat':
            fusion_input_dim = hidden_dim * len(n_ffts)
            self.fusion = nn.Linear(fusion_input_dim, output_dim)
        elif fusion_method == 'add':
            assert all(hidden_dim == hidden_dim for _ in n_ffts), "All hidden dims must be equal for add fusion"
            self.fusion = nn.Linear(hidden_dim, output_dim)
        elif fusion_method == 'attention':
            self.attention_weights = nn.Parameter(torch.ones(len(n_ffts)) / len(n_ffts))
            self.fusion = nn.Linear(hidden_dim, output_dim)
        else:
            raise ValueError(f"Unsupported fusion method: {fusion_method}")
    
    def forward(self, stfts: List[torch.Tensor]) -> torch.Tensor:
        """
        Process multi-resolution STFT features.
        
        Args:
            stfts: List of complex STFT tensors [B, C, F, T']
            
        Returns:
            Fused features [B, output_dim, T']
        """
        features = []
        target_time_steps = None
        
        for i, stft in enumerate(stfts):
            # Convert complex to real/imag channels: [B, C, F, T'] -> [B, C*2, F, T']
            real_imag = torch.stack([stft.real, stft.imag], dim=2)  # [B, C, 2, F, T']
            real_imag = real_imag.reshape(stft.shape[0], -1, stft.shape[2], stft.shape[3])  # [B, C*2, F, T']
            
            # Process with encoder
            feat = self.encoders[i](real_imag)  # [B, hidden_dim, F, T']
            
            # Global average pooling over frequency dimension
            feat = feat.mean(dim=2)  # [B, hidden_dim, T']
            
            # Interpolate to common time dimension (use the longest sequence)
            if target_time_steps is None:
                target_time_steps = max(s.shape[-1] for s in stfts)
            
            if feat.shape[-1] != target_time_steps:
                feat = F.interpolate(feat, size=target_time_steps, mode='linear', align_corners=False)
            
            features.append(feat)
        
        # Fuse features
        if self.fusion_method == 'concat':
            # Concatenate along channel dimension
            fused = torch.cat(features, dim=1)  # [B, hidden_dim*N, T']
            fused = fused.transpose(1, 2)  # [B, T', hidden_dim*N]
            fused = self.fusion(fused)  # [B, T', output_dim]
            fused = fused.transpose(1, 2)  # [B, output_dim, T']
            
        elif self.fusion_method == 'add':
            # Element-wise addition
            fused = torch.stack(features, dim=0).sum(dim=0)  # [B, hidden_dim, T']
            fused = fused.transpose(1, 2)  # [B, T', hidden_dim]
            fused = self.fusion(fused)  # [B, T', output_dim]
            fused = fused.transpose(1, 2)  # [B, output_dim, T']
            
        elif self.fusion_method == 'attention':
            # Attention-weighted fusion
            weights = F.softmax(self.attention_weights, dim=0)
            fused = sum(w * feat for w, feat in zip(weights, features))  # [B, hidden_dim, T']
            fused = fused.transpose(1, 2)  # [B, T', hidden_dim]
            fused = self.fusion(fused)  # [B, T', output_dim]
            fused = fused.transpose(1, 2)  # [B, output_dim, T']
        
        return fused