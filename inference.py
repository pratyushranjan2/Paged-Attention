import torch
from dataset import encode, decode, n_vocab
from components.gpt import Gpt
import time

@torch.no_grad()
def generate(model, prompt, max_new_tokens=300, temperature=1.0, top_k=None):
    model.eval()
    idx = torch.tensor([encode(prompt)], dtype=torch.long, device=device)   # (1, n)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -max_len:]                     # crop to the last max_len tokens
        logits = model(idx_cond, seq_id=0, use_cache=False)[:, -1, :]               # (1, V) last position only
        logits = logits / temperature

        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = float('-inf')  # keep only the top_k logits

        probs = torch.softmax(logits, dim=-1)            # (1, V)
        next_id = torch.multinomial(probs, num_samples=1)   # (1, 1)
        idx = torch.cat([idx, next_id], dim=1)           # (1, n+1)

    model.train()
    return decode(idx[0].tolist())

@torch.no_grad()
def generate_cached(model, prompt, max_new_tokens=300, temperature=1.0, top_k=None):
    model.eval()
    model.reset_cache()

    idx = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    x = idx
    
    prefill = True                                              # prefill with the whole prompt

    for _ in range(max_new_tokens):
        logits = model(x, seq_id=0, use_cache=True, prefill=prefill)[:, -1, :]      # (1, V)
        logits = logits / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = float('-inf')
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)
        x = next_id
        prefill = False                                      # only the new token from now on

    model.reset_cache()
    model.train()
    return decode(idx[0].tolist())

def timed(fn, *args, **kwargs):
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn(*args, **kwargs)
    if device == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - t0

def perf():
    prompt = "thy not"
    n_new = max_len - len(encode(prompt))     # generate as much as fits without hitting max_len
    kw = dict(max_new_tokens=n_new, temperature=0.8, top_k=40)

    # warmup (first call includes CUDA init / kernel compilation overhead)
    generate(model, prompt, max_new_tokens=5)
    generate_cached(model, prompt, max_new_tokens=5)

    t_plain  = timed(generate, model, prompt, **kw)
    t_cached = timed(generate_cached, model, prompt, **kw)

    print(f"no cache: {t_plain:.2f}s  ({n_new / t_plain:.1f} tok/s)")
    print(f"cached:   {t_cached:.2f}s  ({n_new / t_cached:.1f} tok/s)")
    print(f"speedup:  {t_plain / t_cached:.2f}x")




max_len = 256
n_emb = 256
n_head = 8
n_key = 32
n_hidden = 1024
n_decoders = 4
num_blocks = 128
block_size = 4

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Gpt(max_len, n_emb, n_key, n_head, n_hidden, n_decoders, n_vocab, num_blocks=num_blocks, block_size=block_size, device=device).to(device)

perf()