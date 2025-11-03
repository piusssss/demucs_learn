"""
ResUNet++ implementation with improved residual connections and dense skip connections.
Based on "ResUNet++: An Advanced Architecture for Medical Image Segmentation"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class ResidualBlock(nn.Module):
    """
    Residual block with batch normalization and ReLU activation.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        stride: Stride for the first convolution
        downsample: Optional downsampling layer
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None
    ):
        super().__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, 
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation block for channel attention.
    
    Args:
        channels: Number of input channels
        reduction: Reduction ratio for the bottleneck
    """
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Squeeze
        y = self.global_pool(x).view(B, C)
        
        # Excitation
        y = self.fc(y).view(B, C, 1, 1)
        
        # Scale
        return x * y.expand_as(x)


class ResUNetPlusEncoder(nn.Module):
    """
    ResUNet++ Encoder with residual blocks and squeeze-excitation.
    
    Args:
        in_channels: Number of input channels
        base_channels: Base number of channels
        depth: Depth of the encoder
        use_se: Whether to use Squeeze-Excitation blocks
    """
    
    def __init__(
        self,
        in_channels: int = 2,
        base_channels: int = 64,
        depth: int = 4,
        use_se: bool = True
    ):
        super().__init__()
        
        self.depth = depth
        self.use_se = use_se
        
        # Initial convolution
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList()
        self.se_blocks = nn.ModuleList() if use_se else None
        
        channels = base_channels
        for i in range(depth):
            # Determine if we need downsampling
            stride = 2 if i > 0 else 1
            next_channels = channels * 2 if i > 0 else channels
            
            # Downsampling layer if needed
            downsample = None
            if stride != 1 or channels != next_channels:
                downsample = nn.Sequential(
                    nn.Conv2d(channels, next_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(next_channels)
                )
            
            # Residual block
            layer = ResidualBlock(channels, next_channels, stride, downsample)
            self.encoder_layers.append(layer)
            
            # Squeeze-Excitation block
            if use_se:
                se_block = SqueezeExcitation(next_channels)
                self.se_blocks.append(se_block)
            
            channels = next_channels
    
    def forward(self, x):
        """
        Forward pass through encoder.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            List of feature maps from each encoder level
        """
        features = []
        
        # Initial convolution
        x = self.init_conv(x)
        features.append(x)
        
        # Encoder layers
        for i, layer in enumerate(self.encoder_layers):
            x = layer(x)
            
            # Apply SE block if enabled
            if self.use_se:
                x = self.se_blocks[i](x)
            
            features.append(x)
        
        return features


class AttentionGate(nn.Module):
    """
    Attention gate for focusing on relevant features during skip connections.
    
    Args:
        gate_channels: Number of channels in the gating signal
        skip_channels: Number of channels in the skip connection
        inter_channels: Number of intermediate channels
    """
    
    def __init__(self, gate_channels: int, skip_channels: int, inter_channels: int):
        super().__init__()
        
        self.gate_conv = nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=False)
        self.skip_conv = nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=False)
        
        self.relu = nn.ReLU(inplace=True)
        self.attention_conv = nn.Conv2d(inter_channels, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, gate, skip):
        """
        Args:
            gate: Gating signal from deeper layer [B, gate_channels, H_g, W_g]
            skip: Skip connection features [B, skip_channels, H_s, W_s]
            
        Returns:
            Attention-weighted skip features
        """
        # Resize gate to match skip dimensions
        gate_resized = F.interpolate(gate, size=skip.shape[2:], mode='bilinear', align_corners=False)
        
        # Compute attention weights
        gate_feat = self.gate_conv(gate_resized)
        skip_feat = self.skip_conv(skip)
        
        combined = self.relu(gate_feat + skip_feat)
        attention = self.sigmoid(self.attention_conv(combined))
        
        # Apply attention to skip features
        return skip * attention


class ResUNetPlusDecoder(nn.Module):
    """
    ResUNet++ Decoder with attention gates and dense skip connections.
    
    Args:
        encoder_channels: List of channel numbers from encoder
        base_channels: Base number of channels
        use_attention: Whether to use attention gates
    """
    
    def __init__(
        self,
        encoder_channels: List[int],
        base_channels: int = 64,
        use_attention: bool = True
    ):
        super().__init__()
        
        self.use_attention = use_attention
        self.encoder_channels = encoder_channels
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList()
        self.attention_gates = nn.ModuleList() if use_attention else None
        
        # Build decoder from bottom to top
        for i in range(len(encoder_channels) - 1, 0, -1):
            in_channels = encoder_channels[i]
            skip_channels = encoder_channels[i - 1]
            out_channels = skip_channels
            
            # Attention gate
            if use_attention:
                inter_channels = min(in_channels, skip_channels) // 2
                inter_channels = max(inter_channels, 1)  # Ensure at least 1 channel
                attention_gate = AttentionGate(
                    gate_channels=in_channels,
                    skip_channels=skip_channels,
                    inter_channels=inter_channels
                )
                self.attention_gates.append(attention_gate)
            
            # Decoder block - simplified to avoid channel mismatch
            decoder_block = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            self.decoder_layers.append(decoder_block)
    
    def forward(self, encoder_features):
        """
        Forward pass through decoder.
        
        Args:
            encoder_features: List of feature maps from encoder
            
        Returns:
            Decoded feature map
        """
        x = encoder_features[-1]  # Start from the deepest features
        
        for i, decoder_layer in enumerate(self.decoder_layers):
            # Get corresponding skip connection
            skip_idx = len(encoder_features) - 2 - i
            skip = encoder_features[skip_idx]
            
            # Upsample current features
            x = decoder_layer[0](x)  # ConvTranspose2d
            x = decoder_layer[1](x)  # BatchNorm2d
            x = decoder_layer[2](x)  # ReLU
            
            # Apply attention gate if enabled
            if self.use_attention:
                skip = self.attention_gates[i](x, skip)
            
            # Concatenate with skip connection
            x = torch.cat([x, skip], dim=1)
            
            # Apply conv block
            x = decoder_layer[3](x)  # Conv2d
            x = decoder_layer[4](x)  # BatchNorm2d  
            x = decoder_layer[5](x)  # ReLU
        
        return x


class ResUNetPlus(nn.Module):
    """
    Complete ResUNet++ architecture.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        base_channels: Base number of channels
        depth: Depth of the network
        use_se: Whether to use Squeeze-Excitation blocks
        use_attention: Whether to use attention gates
    """
    
    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        base_channels: int = 64,
        depth: int = 4,
        use_se: bool = True,
        use_attention: bool = True
    ):
        super().__init__()
        
        # Encoder
        self.encoder = ResUNetPlusEncoder(
            in_channels=in_channels,
            base_channels=base_channels,
            depth=depth,
            use_se=use_se
        )
        
        # Calculate encoder channel dimensions
        encoder_channels = [base_channels]
        channels = base_channels
        for i in range(depth):
            if i > 0:
                channels *= 2
            encoder_channels.append(channels)
        
        # Decoder
        self.decoder = ResUNetPlusDecoder(
            encoder_channels=encoder_channels,
            base_channels=base_channels,
            use_attention=use_attention
        )
        
        # Final output layer
        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        """
        Forward pass through ResUNet++.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Output tensor [B, out_channels, H, W]
        """
        # Encoder
        encoder_features = self.encoder(x)
        
        # Decoder
        x = self.decoder(encoder_features)
        
        # Final output
        x = self.final_conv(x)
        
        return x