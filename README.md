# Mini GPT with Paged Attention

A small GPT-style decoder-only Transformer, written from scratch for learning, whose self-attention layer stores its KV cache in a **paged** layout (fixed-size blocks, a block table per sequence, and a free-list allocator) instead of one contiguous tensor per sequence. The repo also includes model-free benchmarks that measure how much KV memory paging saves compared with a preallocated contiguous cache.

## What's in here

```
components/
  gpt.py                  # token + positional embedding, stack of decoders, output head
  decoder.py              # attention + feed-forward block with residuals and LayerNorm
  multi_head_attention.py # self-attention that reads and writes the paged KV cache
  feed_forward.py
  positional_embedding.py
paged_cache/
  block_allocator.py      # free list of physical block ids
  paged_kv_cache.py       # block pools, block tables, reserve / write / gather / free
inference.py              # generation with and without the cache, plus a timing check
bench_memory.py           # memory benchmarks (no model needed)
```

Default model config (`inference.py`): 4 decoder layers, 8 heads, head dim 32, embedding 256, `max_len` 256, paged cache with `block_size = 4` and `num_blocks = 128`.

## How the paged cache works

| Piece | What it is |
|---|---|
| Block pool | One K tensor and one V tensor per layer, shaped `(num_blocks, block_size, n_kv_heads, head_dim)`. Allocated once. |
| `BlockAllocator` | A free list of block ids. `allocate()` and `release()` are pure bookkeeping, with no memory calls. |
| Block table | Per sequence, a list mapping logical block index to physical block id. One table is shared by all layers. |

A token at position `pos` lives at logical block `pos // block_size`, slot `pos % block_size`, and the block table translates the logical block to a physical one.

Per generated token, the model does:

1. `reserve(seq_id)`, once per token. It allocates a new block when the token starts one, advances the sequence length, and returns `(physical_block, slot)`.
2. `write(layer, physical, slot, k, v)`, once per layer. It copies K and V into the pool.
3. `gather(layer, seq_id)`, once per layer during decode. It collects the sequence's blocks in logical order and trims to the real length, so standard attention can run on the result.
4. `free(seq_id)` when the sequence finishes, returning its blocks to the free list.

Prefill processes the whole prompt at once and decode adds one token per step. In both cases `reserve` runs once per token, not once per layer.

## Benchmarks

`bench_memory.py` uses the real `PagedKVCache` for the paged bookkeeping and never runs the model. The contiguous baseline reserves `max_len` slots per request, which is what a naive server does. Request length means prompt plus generated tokens.

Setup: `max_len = 256`, `block_size = 4`, 8 KB of KV per token (4 layers, 8 heads, head dim 32, fp32). Three length distributions: `lognormal` (skewed, median about 40 tokens), `uniform` (8 to 256), and `near_max` (80% to 100% of `max_len`, the worst case for paging).

```
python bench_memory.py            # figures and results.json go to bench_out/
```

### 1. Utilization (used slots / reserved slots)

| Distribution | Contiguous | Paged |
|---|---|---|
| lognormal | 21% | 97% |
| uniform | 52% | 99% |
| near_max | 90% | 99% |

Contiguous utilization is the mean request length divided by `max_len`, so it falls as requests get shorter. Paged utilization stays near 100% because it only wastes the unused tail of each sequence's last block.

### 2. Wasted slots per sequence

| Scheme | Wasted slots per sequence |
|---|---|
| Contiguous | `max_len - len` (up to 255) |
| Paged, block size 4 | at most 3 |
| Paged, block size 16 | at most 15 |
| Paged, block size 64 | at most 63 |

Paged waste is bounded by `block_size - 1` regardless of `max_len`, and it shows up as a sawtooth that resets at each block boundary. A larger block size means fewer table entries but more waste.

### 3. Max concurrent sequences under a fixed KV budget

Requests are admitted first-come-first-served until the pool runs out. Paged values are means over 30 random request streams, read off the plot, so treat them as approximate. Contiguous is exactly `budget // max_len`.

| Distribution | Budget (token slots) | Contiguous | Paged (approx.) | Gain |
|---|---|---|---|---|
| lognormal | 512 | 2 | ~8 | ~4x |
| lognormal | 16384 | 64 | ~280 | ~4.4x |
| uniform | 512 | 2 | ~4 | ~2x |
| uniform | 16384 | 64 | ~120 | ~1.9x |
| near_max | 512 | 2 | 2 | 1x |
| near_max | 16384 | 64 | ~70 | ~1.1x |

The gain depends on the gap between typical request length and `max_len`. When requests are close to `max_len` there is nothing to reclaim, and paging barely helps.

The script also includes a block-size sweep (experiment 4) and timings for `reserve()` and `gather()` (experiment 5), which are not shown above.

## References

- Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention" (SOSP 2023).
- Yu et al., "Orca: A Distributed Serving System for Transformer-Based Generative Models" (OSDI 2022), for iteration-level scheduling.
