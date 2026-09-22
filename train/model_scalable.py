from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ScaledTinyUNet(nn.Module):
    """Same 3-level U-Net topology with configurable channel width.

    Input remains RGB and output remains RGB. Increasing base_channels
    scales [16,32,64,128] proportionally without changing the U-Net topology.
    """

    def __init__(
        self,
        input_size: int = 512,
        base_channels: int = 16,
        activation_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if input_size <= 0:
            raise ValueError("input_size must be positive")
        if input_size % 8 != 0:
            raise ValueError("input_size must be divisible by 8")
        if base_channels <= 0:
            raise ValueError("base_channels must be positive")

        c1, c2, c3, c4 = (
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
        )

        self.input_size = input_size
        self.base_channels = base_channels
        self.channels = (c1, c2, c3, c4)
        self.activation_checkpointing = activation_checkpointing

        self.enc1 = ConvBlock(3, c1)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(c1, c2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(c2, c3)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(c3, c4)

        self.up3 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec3 = ConvBlock(c4 + c3, c3)

        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec2 = ConvBlock(c3 + c2, c2)

        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec1 = ConvBlock(c2 + c1, c1)

        self.output = nn.Conv2d(c1, 3, kernel_size=1, bias=True)

        self._initialize_like_tfjs()

    @staticmethod
    def _initialize_conv(layer: nn.Conv2d) -> None:
        nn.init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)

    def _initialize_like_tfjs(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                self._initialize_conv(module)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _run_block(self, block: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.activation_checkpointing and torch.is_grad_enabled():
            return checkpoint(block, x, use_reentrant=False)
        return block(x)

    @staticmethod
    def _cat(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.cat([a, b], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self._run_block(self.enc1, x)
        p1 = self.pool1(e1)

        e2 = self._run_block(self.enc2, p1)
        p2 = self.pool2(e2)

        e3 = self._run_block(self.enc3, p2)
        p3 = self.pool3(e3)

        b = self._run_block(self.bottleneck, p3)

        u3 = self.up3(b)
        d3 = self._run_block(self.dec3, self._cat(u3, e3))

        u2 = self.up2(d3)
        d2 = self._run_block(self.dec2, self._cat(u2, e2))

        u1 = self.up1(d2)
        d1 = self._run_block(self.dec1, self._cat(u1, e1))

        return torch.sigmoid(self.output(d1))


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def describe(model: ScaledTinyUNet) -> dict:
    return {
        "input_size": model.input_size,
        "base_channels": model.base_channels,
        "channels": list(model.channels),
        "activation_checkpointing": model.activation_checkpointing,
        "trainable_params": parameter_count(model),
        "state_values": sum(t.numel() for t in model.state_dict().values()),
    }
