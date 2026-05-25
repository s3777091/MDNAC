from __future__ import annotations

from collections.abc import Mapping
import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, cfg: Mapping[str, object]) -> None:
        super().__init__()
        dtype = cfg["dtype"]
        emb_dim = int(cfg["emb_dim"])
        hidden_dim = int(cfg["hidden_dim"])
        self.fc1 = nn.Linear(emb_dim, hidden_dim, dtype=dtype, bias=False)
        self.fc2 = nn.Linear(emb_dim, hidden_dim, dtype=dtype, bias=False)
        self.fc3 = nn.Linear(hidden_dim, emb_dim, dtype=dtype, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = nn.functional.silu(x_fc1) * x_fc2
        return self.fc3(x)
