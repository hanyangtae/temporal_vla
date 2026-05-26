"""SAFE-LSTM detector definition + load + causal inference helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class LSTMDetector(nn.Module):
    """Single-layer LSTM + linear head with sigmoid output (SAFE-LSTM)."""

    def __init__(self, input_dim: int = 1024, hidden_dim: int = 256):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return torch.sigmoid(self.fc(out)).squeeze(-1)


def load_detector(ckpt_path: Path, hidden_dim: int = 256, device: str = "cpu") -> LSTMDetector:
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    input_dim = state["lstm.weight_ih_l0"].shape[1]
    model = LSTMDetector(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def lstm_hidden_sequence(model: LSTMDetector, X: np.ndarray, device: str) -> np.ndarray:
    """Return the per-timestep LSTM hidden state, shape [T, hidden_dim].

    LSTM is causal: out[t] only depends on x[:t+1]."""
    x = torch.from_numpy(X).float().unsqueeze(0).to(device)
    out, _ = model.lstm(x)
    return out.squeeze(0).cpu().numpy()


@torch.no_grad()
def lstm_score_sequence(model: LSTMDetector, X: np.ndarray, device: str) -> np.ndarray:
    """Return the per-timestep failure probability, shape [T]."""
    x = torch.from_numpy(X).float().unsqueeze(0).to(device)
    return model(x).squeeze(0).cpu().numpy()
