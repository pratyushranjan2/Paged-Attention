import torch

from paged_cache.block_allocator import BlockAllocator

class PagedKVCache:
    def __init__(self, num_layers, num_blocks, block_size, n_kv_heads, head_dim, device="cpu"):
        self.block_size = block_size
        self.allocator = BlockAllocator(num_blocks)
        self.k_pools = [torch.zeros(num_blocks, block_size, n_kv_heads, head_dim, device=device) for _ in range(num_layers)]
        self.v_pools = [torch.zeros(num_blocks, block_size, n_kv_heads, head_dim, device=device) for _ in range(num_layers)]
        self.block_tables = {}
        self.seq_lens = {}
    
    def add_sequence(self, seq_id):
        self.block_tables[seq_id] = []
        self.seq_lens[seq_id] = 0

    def reserve(self, seq_id):
        pos = self.seq_lens[seq_id]
        logical = pos // self.block_size
        if logical == len(self.block_tables[seq_id]):
            self.block_tables[seq_id].append(self.allocator.allocate())
        slot = pos % self.block_size
        self.seq_lens[seq_id] += 1
        return self.block_tables[seq_id][logical], slot

    def write(self, layer_idx, physical, slot, k_new, v_new):
        self.k_pools[layer_idx][physical, slot] = k_new
        self.v_pools[layer_idx][physical, slot] = v_new

    def gather(self, layer_id, seq_id):
        physicals = self.block_tables[seq_id]
        length = self.seq_lens[seq_id]
        k = self.k_pools[layer_id][physicals].flatten(0,1)[:length]
        v = self.v_pools[layer_id][physicals].flatten(0,1)[:length]
        return k, v

    def free(self, seq_id):
        for block_id in self.block_tables[seq_id]:
            self.allocator.release(block_id)
        self.block_tables.pop(seq_id, None)
        self.seq_lens.pop(seq_id, None)