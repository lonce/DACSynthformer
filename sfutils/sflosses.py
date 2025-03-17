
import torch
import torch.nn.functional as F


# The goal is to do better than the MSE loss
# Use multi-scale, and improve glitchy output.

# Applies L1 or MSE loss at multiple scales.
# Ensures both frame-level accuracy and long-term stability.
# Balances fine and coarse structure, reducing temporal artifacts.


def downsample_mel(mel_spec, scales=[1, 2, 4, 8]):
    """Downsamples the mel spectrogram along the time axis using average pooling.
       Scales define the time resolution levels to be considered."""
    mel_scales = []
    for scale in scales:
        if scale == 1:
            mel_scales.append(mel_spec)  # Original scale
        else:
            mel_scales.append(F.avg_pool2d(mel_spec, kernel_size=(1, scale), stride=(1, scale)))
    return mel_scales


def multi_scale_mel_loss(pred_mel, target_mel, scales=[1, 2, 4, 8], loss_type="l1"):
    pred_mels = downsample_mel(pred_mel, scales)
    target_mels = downsample_mel(target_mel, scales)

    loss = 0
    for pred, target in zip(pred_mels, target_mels):
        if loss_type == "l1":
            loss += torch.mean(torch.abs(pred - target))  # L1 loss
        else:
            loss += torch.mean((pred - target) ** 2)  # MSE loss
        
    return loss / len(scales)  # Normalize across scales
