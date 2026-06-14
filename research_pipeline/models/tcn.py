from __future__ import annotations

from dataclasses import dataclass

from research_pipeline.utils.errors import DependencyMissingError


def require_torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise DependencyMissingError(
            "PyTorch is required for the production C1-T TCN. "
            "Install requirements/windows-train.txt in a Python 3.11/3.12 environment."
        ) from exc
    return torch, nn


@dataclass(frozen=True)
class TCNConfig:
    input_dim: int = 74
    num_classes: int = 7
    channels: tuple[int, ...] = (64, 64, 96)
    kernel_size: int = 3
    dropout: float = 0.15
    pooling: str = "avg"


def build_tcn(config: TCNConfig):
    torch, nn = require_torch()

    class TemporalBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, dilation: int):
            super().__init__()
            padding = dilation * (config.kernel_size - 1) // 2
            self.net = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, config.kernel_size, padding=padding, dilation=dilation),
                nn.BatchNorm1d(out_channels),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Conv1d(out_channels, out_channels, config.kernel_size, padding=padding, dilation=dilation),
                nn.BatchNorm1d(out_channels),
                nn.GELU(),
            )
            self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

        def forward(self, x):
            return self.net(x) + self.skip(x)

    class CompactTCN(nn.Module):
        def __init__(self):
            super().__init__()
            blocks = []
            in_channels = config.input_dim
            for index, out_channels in enumerate(config.channels):
                blocks.append(TemporalBlock(in_channels, out_channels, dilation=2**index))
                in_channels = out_channels
            self.encoder = nn.Sequential(*blocks)
            if config.pooling == "avgmax":
                self.head = nn.Linear(in_channels * 2, config.num_classes)
            else:
                self.head = nn.Sequential(
                    nn.AdaptiveAvgPool1d(1),
                    nn.Flatten(),
                    nn.Linear(in_channels, config.num_classes),
                )

        def forward(self, x):
            # x: [B,T,F]
            x = x.transpose(1, 2)
            encoded = self.encoder(x)
            if config.pooling == "avgmax":
                pooled = torch.mean(encoded, dim=2)
                pooled = torch.cat([pooled, torch.amax(encoded, dim=2)], dim=1)
                return self.head(pooled)
            return self.head(encoded)

    return CompactTCN()
