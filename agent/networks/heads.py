"""Production, worker, and value-head primitives."""

from torch import nn


def make_scalar_head(
    input_dim: int, hidden_dim: int, dropout: float
) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.Tanh(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, 1),
    )
