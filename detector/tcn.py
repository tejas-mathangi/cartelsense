"""
tcn.py — Temporal Convolutional Network for collusion detection.

Architecture: 3 TCN blocks with dilated causal convolutions.

Why TCN over LSTM?
  - Parallelizable: all timesteps computed simultaneously (faster on CPU)
  - Explicit receptive field: dilation factors 1,2,4 → sees 7 steps back per block
  - No vanishing gradient through time
  - Empirically matches LSTM on sequence classification tasks

Input:  (batch, n_features, window_size) — channels-first format
Output: (batch, 1) — collusion probability per window
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """
    A causal (left-padded) dilated 1D convolution.

    "Causal" means: output at time T only depends on inputs at times ≤ T.
    We cannot look into the future when making a detection decision.

    Dilation = D means the kernel sees timesteps [T, T-D, T-2D, ...].
    With kernel_size=3 and dilation=D, receptive field = 1 + 2*D timesteps.

    Implementation: pad the LEFT side of the sequence by (kernel_size-1)*dilation
    zeros, then apply a standard Conv1d. This ensures no future leakage.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        self.padding   = (kernel_size - 1) * dilation
        self.conv      = nn.Conv1d(
            in_channels, out_channels,
            kernel_size = kernel_size,
            dilation    = dilation,
            padding     = 0,   # we handle padding manually
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, time)
        # Pad left with zeros (causal padding)
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


class TCNBlock(nn.Module):
    """
    One TCN residual block.

    Structure:
      CausalConv1d → LayerNorm → ReLU → Dropout
      CausalConv1d → LayerNorm → ReLU → Dropout
      + residual connection (with 1×1 conv if channel dims differ)

    The residual connection lets gradients flow directly to earlier layers,
    preventing vanishing gradients even with many blocks.

    LayerNorm over the channel dimension (not BatchNorm) because:
      - Works well with small batch sizes
      - More stable during inference
      - Better for time series with varying statistics
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, dilation: int = 1, dropout: float = 0.2):
        super().__init__()

        self.conv1    = CausalConv1d(in_channels,  out_channels, kernel_size, dilation)
        self.conv2    = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.norm1    = nn.LayerNorm(out_channels)
        self.norm2    = nn.LayerNorm(out_channels)
        self.dropout  = nn.Dropout(dropout)
        self.relu     = nn.ReLU()

        # 1×1 conv to match dimensions for residual if needed
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual_conv(x)

        # First conv layer
        out = self.conv1(x)
        # LayerNorm expects (batch, time, channels) — transpose, norm, transpose back
        out = self.norm1(out.transpose(1, 2)).transpose(1, 2)
        out = self.relu(out)
        out = self.dropout(out)

        # Second conv layer
        out = self.conv2(out)
        out = self.norm2(out.transpose(1, 2)).transpose(1, 2)
        out = self.relu(out)
        out = self.dropout(out)

        # Residual connection
        return self.relu(out + residual)


class CollusionTCN(nn.Module):
    """
    Full TCN classifier for collusion detection.

    Architecture:
      Input → TCNBlock(dilation=1) → TCNBlock(dilation=2) → TCNBlock(dilation=4)
           → Global Average Pool → FC(64) → ReLU → Dropout → FC(1) → Sigmoid

    Receptive field: with kernel_size=3 and dilations [1,2,4]:
      Each block sees: 1 + 2*dilation*2 timesteps = 3,5,9 per conv pair
      Combined: approximately 17 timesteps of context
      With 24-timestep windows this covers most of the window.

    Global Average Pool collapses the time dimension, making the classifier
    invariant to where in the window the collusion signal appears.

    Parameters
    ----------
    n_features   : number of input feature channels (13)
    hidden_dim   : channels in TCN blocks (default 32 — small for CPU)
    kernel_size  : convolution kernel size
    dropout      : dropout probability during training
    """

    def __init__(
        self,
        n_features:  int   = 13,
        hidden_dim:  int   = 32,
        kernel_size: int   = 3,
        dropout:     float = 0.2,
    ):
        super().__init__()

        self.n_features = n_features
        self.hidden_dim = hidden_dim

        # Three TCN blocks — dilation doubles each block
        # Receptive field grows: 3 → 5 → 9 timesteps per block
        self.tcn_blocks = nn.Sequential(
            TCNBlock(n_features,  hidden_dim, kernel_size, dilation=1, dropout=dropout),
            TCNBlock(hidden_dim,  hidden_dim, kernel_size, dilation=2, dropout=dropout),
            TCNBlock(hidden_dim,  hidden_dim, kernel_size, dilation=4, dropout=dropout),
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, n_features, window_size)
        returns: (batch,) — collusion probability in [0, 1]
        """
        # TCN blocks: (batch, n_features, T) → (batch, hidden_dim, T)
        out = self.tcn_blocks(x)

        # Global average pooling over time: (batch, hidden_dim, T) → (batch, hidden_dim)
        out = out.mean(dim=2)

        # Classifier: (batch, hidden_dim) → (batch, 1)
        out = self.classifier(out)

        # Sigmoid to get probability
        return torch.sigmoid(out).squeeze(1)   # (batch,)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)