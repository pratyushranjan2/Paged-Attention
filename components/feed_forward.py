import torch
import torch.nn as nn

class FeedForward(nn.Module):
  
  def __init__(self, n_emb, n_hidden):
    super().__init__()
    self.lin1 = nn.Linear(n_emb, n_hidden)
    self.lin2 = nn.Linear(n_hidden, n_emb)
  
  def forward(self, x):
    x = self.lin1(x)
    x = torch.relu(x)
    x = self.lin2(x)
    return x