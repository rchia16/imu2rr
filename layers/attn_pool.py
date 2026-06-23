import torch
import torch.nn as nn
import torch.nn.functional as F

class AttnPool1D(nn.Module):
    """
    Attention pooling over time.
    x: (B, T, D) -> pooled: (B, D)
    """
    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.q = nn.Parameter(torch.randn(d_model))          # learned query vector
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # scores: (B, T)
        scores = torch.einsum("btd,d->bt", x, self.q)
        w = F.softmax(scores, dim=1)
        w = self.dropout(w)
        # pooled: (B, D)
        pooled = torch.einsum("btd,bt->bd", x, w)
        return pooled

