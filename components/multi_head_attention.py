import torch
import torch.nn as nn
from math import sqrt

class MultiHeadSelfAttention(nn.Module):
  
  def __init__(self, n_emb, n_key, n_head, layer_id, kv_cache):
    super().__init__()
    self.n_key = n_key
    self.n_head = n_head
    self.Qlin = nn.Linear(n_emb, n_head*n_key)
    self.Klin = nn.Linear(n_emb, n_head*n_key)
    self.Vlin = nn.Linear(n_emb, n_head*n_key)
    self.W0 = nn.Linear(n_head*n_key, n_emb)
    self.layer_id = layer_id
    self.kv_cache = kv_cache
  
  def forward(self, x, seq_id, slots, use_cache = False, prefill = False):
    Q = self.Qlin(x) # (b,n,h*k)
    K = self.Klin(x)
    V = self.Vlin(x)

    Q = Q.view(Q.shape[0], Q.shape[1], self.n_head, self.n_key).transpose(1,2) # (b,h,n,k)
    K  = K.view(K.shape[0], K.shape[1], self.n_head, self.n_key) # (b,n,h,k)
    V = V.view(V.shape[0], V.shape[1], self.n_head, self.n_key) # (b,n,h,k)

    if use_cache:
      # save new key and value in the cache
      for idx, (physical, slot) in enumerate(slots):
        self.kv_cache.write(self.layer_id, physical, slot, K[:,idx,:,:].squeeze(0), V[:,idx,:,:].squeeze(0))
      
      # construct full key and value set using cached values
      if not prefill:
          K, V = self.kv_cache.gather(self.layer_id, seq_id) # (n,h,k)
          K = K.unsqueeze(0) # (b,n,h,k)
          V = V.unsqueeze(0)

    K = K.transpose(1,2) # (b, h, n, k)
    V = V.transpose(1,2)
    score = Q @ K.transpose(-2,-1) # (b,h,n,n), inf: (b,h,1,n)
    score = score / sqrt(self.n_key)
    n_new, n_total = score.shape[-2], score.shape[-1]
    mask = torch.tril(torch.ones(n_total, n_total, device=score.device))[-n_new:]
    score = score.masked_fill(mask == 0, float('-inf'))
    score = torch.softmax(score, -1)

    Z = score @ V # (b,h,n,k)
    Z = Z.transpose(1,2) # (b,n,h,k)
    Z = Z.reshape(Z.shape[0], Z.shape[1], -1) # (b,n,h*k)

    Z = self.W0(Z) # (b,n,e)

    return Z

  def reset_cache(self):
    pass