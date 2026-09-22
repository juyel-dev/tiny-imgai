from __future__ import annotations

import torch
from torch import nn


def luminance(x: torch.Tensor) -> torch.Tensor:
    return (
        0.2126 * x[:, 0:1]
        + 0.7152 * x[:, 1:2]
        + 0.0722 * x[:, 2:3]
    )


class PrintCleanLoss(nn.Module):
    """Loss for black-background/colored notes -> white-background black notes.

    The model architecture stays unchanged. This objective gives extra
    importance to foreground pixels, white background, grayscale output,
    and local edges so the network is not rewarded for producing a blurry
    gray compromise.
    """

    def __init__(
        self,
        foreground_threshold: float = 0.95,
        foreground_weight: float = 5.0,
        pixel_weight: float = 0.55,
        mask_weight: float = 0.20,
        background_weight: float = 0.10,
        edge_weight: float = 0.10,
        grayscale_weight: float = 0.05,
    ) -> None:
        super().__init__()
        self.foreground_threshold = foreground_threshold
        self.foreground_weight = foreground_weight
        self.pixel_weight = pixel_weight
        self.mask_weight = mask_weight
        self.background_weight = background_weight
        self.edge_weight = edge_weight
        self.grayscale_weight = grayscale_weight

    @staticmethod
    def _edge_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_dx = pred[..., :, 1:] - pred[..., :, :-1]
        pred_dy = pred[..., 1:, :] - pred[..., :-1, :]
        target_dx = target[..., :, 1:] - target[..., :, :-1]
        target_dy = target[..., 1:, :] - target[..., :-1, :]
        return 0.5 * (
            torch.mean(torch.abs(pred_dx - target_dx))
            + torch.mean(torch.abs(pred_dy - target_dy))
        )

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        pred_luma = luminance(prediction)
        target_luma = luminance(target)

        # Target foreground is dark content; the rest should become white.
        foreground = (target_luma < self.foreground_threshold).float()
        weights = 1.0 + self.foreground_weight * foreground

        pixel_error = torch.abs(prediction - target)
        pixel_loss = torch.sum(pixel_error * weights.expand_as(pixel_error)) / (
            torch.sum(weights) * 3.0
        )

        # Predict a binary "ink/content" field from luminance.
        pred_dark = (1.0 - pred_luma).clamp(1e-5, 1.0 - 1e-5)
        mask_target = foreground
        mask_loss = torch.nn.functional.binary_cross_entropy(
            pred_dark,
            mask_target,
        )

        # Force target-white regions toward actual white, rather than gray.
        background = 1.0 - foreground
        background_error = torch.abs(1.0 - pred_luma)
        background_loss = (
            torch.sum(background_error * background)
            / torch.sum(background).clamp_min(1.0)
        )

        # Preserve thin handwriting / diagram boundaries.
        edge_loss = self._edge_loss(pred_luma, target_luma)

        # Remove colored/cyan residuals. Target pages are effectively
        # monochrome after processing.
        mean_pred = pred_luma.expand_as(prediction)
        grayscale_loss = torch.mean(torch.abs(prediction - mean_pred))

        return (
            self.pixel_weight * pixel_loss
            + self.mask_weight * mask_loss
            + self.background_weight * background_loss
            + self.edge_weight * edge_loss
            + self.grayscale_weight * grayscale_loss
        )
