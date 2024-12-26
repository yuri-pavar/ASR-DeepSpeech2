import torch_audiomentations
from torch import Tensor, nn


class ColoredNoise(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._aug = torch_audiomentations.AddColoredNoise(*args, **kwargs)

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)