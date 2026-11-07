import torch
import torch.nn as nn
from components.multi_head_attention import MultiHeadSelfAttention
from components.feed_forward import FeedForward

class Decoder(nn.Module):

  def __init__(self, n_emb, n_key, n_head, n_hidden, kv_cache, layer_id):
    super().__init__()
    self.layer_id = layer_id
    self.kv_cache = kv_cache
    self.attn = MultiHeadSelfAttention(n_emb, n_key, n_head, layer_id, kv_cache)
    self.ff = FeedForward(n_emb, n_hidden)
    self.ln1 = nn.LayerNorm(n_emb)
    self.ln2 = nn.LayerNorm(n_emb)
  
  def forward(self, x, seq_id, slots, use_cache = False, prefill = False):
    # x: (b, n, e)
    x = self.ln1(x + self.attn(x, seq_id, slots, use_cache, prefill))
    x = self.ln2(x + self.ff(x))
    return x
  
  def reset_cache(self):
    self.attn.reset_cache()