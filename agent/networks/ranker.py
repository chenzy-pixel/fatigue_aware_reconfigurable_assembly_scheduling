"""V7 monotone ranker parameter container.

The bounding operation depends on the legal candidate set, so it remains in
the policy forward pass; this class owns the checkpoint-compatible weight.
"""

from torch import nn


class BoundedRanker(nn.Linear):
    def __init__(self, feature_count: int) -> None:
        super().__init__(feature_count, 1, bias=False)
