from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

import torch.nn as nn


class TokenMixerBuilder(Protocol):
    def __call__(self, cfg: Mapping[str, object], layer_type: str, layer_idx: int) -> nn.Module:
        ...
