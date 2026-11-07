import torch
import torch.nn as nn

class PositionalEmbedding(nn.Module):

  def __init__(self, max_len, n_emb):
    super().__init__()
    self.emb = nn.Embedding(max_len, n_emb)
  
  def forward(self, x):
    idx = torch.arange(x.shape[1], device=x.device)
    x = x + self.emb(idx)
    return x