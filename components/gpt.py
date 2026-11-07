import torch
import torch.nn as nn
from components.decoder import Decoder
from components.positional_embedding import PositionalEmbedding
from paged_cache.paged_kv_cache import PagedKVCache

class Gpt(nn.Module):

  def __init__(self, max_len, n_emb, n_key, n_head, n_hidden, n_decoders, n_vocab, num_blocks, block_size, device="cpu"):
    super().__init__()
    self.token_embedding = nn.Embedding(n_vocab, n_emb)
    self.positional_embedding = PositionalEmbedding(max_len, n_emb)
    self.kv_cache = PagedKVCache(num_layers=n_decoders, num_blocks=num_blocks, block_size=block_size, n_kv_heads=n_head, head_dim=n_key, device=device)
    self.decoders = nn.ModuleList(
        [Decoder(n_emb, n_key, n_head, n_hidden, self.kv_cache, layer_id) for layer_id in range(n_decoders)]
    )
    self.linear = nn.Linear(n_emb, n_vocab)
  
  def forward(self, x, seq_id, use_cache = False, prefill = False):
    x = self.token_embedding(x)
    x = self.positional_embedding(x)
    
    if use_cache:
      
      if prefill:
        self.kv_cache.add_sequence(seq_id)
      
      n = x.shape[1] # number of tokens in prefill
      slots = [self.kv_cache.reserve(seq_id) for _ in range(n)]
    
    for decoder in self.decoders:
      if use_cache:
        x = decoder(x, seq_id, slots, use_cache, prefill)
      else:
        x = decoder(x, seq_id, None)
    
    x = self.linear(x)
    return x
  
  def reset_cache(self):
    for decoder in self.decoders:
      decoder.reset_cache()