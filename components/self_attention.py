import torch
import torch.nn as nn
from math import sqrt

class SelfAttention(nn.Module):
  
  def __init__(self, n_emb, n_key):
    super().__init__()
    self.Qlin = nn.Linear(n_emb, n_key)
    self.Klin = nn.Linear(n_emb, n_key)
    self.Vlin = nn.Linear(n_emb, n_key)
    self.n_key = n_key
  
  def forward(self, x):
    # x: (b, n, 4)
    Q = self.Qlin(x) # (b,n,3)
    K = self.Klin(x) # (b,n,3)
    V = self.Vlin(x) # (b,n,3)

    score = Q @ K.transpose(-2,-1) # (b,n,n)
    score = score / sqrt(self.n_key)
    score = torch.softmax(score, -1)

    Z = score @ V # (b,n,3)

    return Z