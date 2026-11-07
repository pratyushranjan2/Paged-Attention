"""
Model-free memory benchmarks: contiguous (preallocated) KV cache vs paged KV cache.

Uses your real PagedKVCache / BlockAllocator, but only calls add_sequence, reserve,
free (pure bookkeeping), so no model or attention is needed. The one exception is
experiment 5, which times gather() on realistic tensor sizes.

Experiments
  1. Utilization (used slots / reserved slots) for several length distributions
  2. Wasted slots per sequence vs sequence length
  3. Max concurrent sequences under a fixed KV memory budget   <- headline
  4. Block size sweep: utilization, capacity, block-table length
  5. Timings: reserve() cost per token, gather() cost vs a contiguous slice

Run from your project root (the folder containing paged_cache/):
    python bench_memory.py
    python bench_memory.py --max-len 256 --n-requests 1000 --trials 30 --out bench_out
"""
import argparse
import json
import math
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from paged_cache.paged_kv_cache import PagedKVCache

# ----------------------------------------------------------------------------
# Config (defaults match your model: 4 layers, 8 heads, head_dim 32, fp32)
# ----------------------------------------------------------------------------
N_LAYERS, N_HEADS, HEAD_DIM, DTYPE_BYTES = 4, 8, 32, 4
BYTES_PER_TOKEN = 2 * N_LAYERS * N_HEADS * HEAD_DIM * DTYPE_BYTES  # K and V, all layers

DISTS = ["lognormal", "uniform", "near_max"]
COLORS = {"contiguous": "#d95f02", "paged": "#1b9e77", "oracle": "#7570b3"}


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def make_cache(num_blocks, block_size, n_kv_heads=1, head_dim=1):
    """Bookkeeping-only cache: tiny tensors, because reserve() never touches the data.
    (Block counts and free-list behaviour don't depend on n_kv_heads/head_dim.)"""
    return PagedKVCache(1, num_blocks, block_size, n_kv_heads, head_dim)


def sample_lengths(dist, n, rng, min_len, max_len):
    if dist == "lognormal":      # skewed: median ~40, a few long ones
        x = np.exp(rng.normal(math.log(40), 0.9, size=n))
        return np.clip(x.astype(int), min_len, max_len)
    if dist == "uniform":
        return rng.integers(min_len, max_len + 1, size=n)
    if dist == "near_max":       # worst case for paging: everything is long
        return rng.integers(int(0.8 * max_len), max_len + 1, size=n)
    raise ValueError(dist)


def fill(cache, seq_id, length):
    """Register a sequence and reserve `length` token slots. Returns #blocks used."""
    cache.add_sequence(seq_id)
    for _ in range(length):
        cache.reserve(seq_id)
    return len(cache.block_tables[seq_id])


def blocks_used_via_cache(length, block_size, max_len):
    """Run one sequence through the real cache and return how many blocks it took."""
    cache = make_cache(math.ceil(max_len / block_size) + 1, block_size)
    n = fill(cache, 0, length)
    assert n == math.ceil(length / block_size), "block count != ceil(len/B)"
    return n


def utilization(lengths, block_size, max_len):
    """Returns (contiguous_util, paged_util, mean_block_table_len) using the real cache."""
    cache = make_cache(math.ceil(max_len / block_size) + 1, block_size)
    reserved_paged, table_lens = 0, []
    for i, L in enumerate(lengths):
        n = fill(cache, i, int(L))
        assert n == math.ceil(L / block_size)
        reserved_paged += n * block_size
        table_lens.append(n)
        cache.free(i)                                   # exercises free(), keeps pool tiny
    assert len(cache.allocator.free_blocks) == math.ceil(max_len / block_size) + 1
    used = int(np.sum(lengths))
    return used / (len(lengths) * max_len), used / reserved_paged, float(np.mean(table_lens))


def max_concurrent_paged(lengths, budget_slots, block_size):
    """Admit requests FIFO (each reserving its full final length) until the pool is out."""
    num_blocks = budget_slots // block_size
    cache = make_cache(num_blocks, block_size)
    admitted = 0
    for i, L in enumerate(lengths):
        try:
            fill(cache, i, int(L))
        except (IndexError, RuntimeError):              # your allocator raises IndexError when empty
            cache.free(i)                               # roll back the partial sequence
            break
        admitted += 1
    return admitted


def max_concurrent_contiguous(budget_slots, max_len):
    return budget_slots // max_len                      # every request reserves max_len slots


def mb(slots):
    return slots * BYTES_PER_TOKEN / 1e6


def table(title, header, rows):
    print(f"\n{title}")
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(header)]
    print("  " + "  ".join(str(h).ljust(w) for h, w in zip(header, widths)))
    print("  " + "  ".join("-" * w for w in widths))
    for r in rows:
        print("  " + "  ".join(str(c).ljust(w) for c, w in zip(r, widths)))


# ----------------------------------------------------------------------------
# Experiment 1: utilization
# ----------------------------------------------------------------------------
def exp1_utilization(args, rng, out):
    res, rows = {}, []
    for d in DISTS:
        lens = sample_lengths(d, args.n_requests, rng, args.min_len, args.max_len)
        c, p, _ = utilization(lens, args.block_size, args.max_len)
        res[d] = {"mean_len": float(lens.mean()), "contiguous": c, "paged": p}
        rows.append([d, f"{lens.mean():.1f}", f"{c:.1%}", f"{p:.1%}"])
    table(f"[1] Utilization (block_size={args.block_size}, max_len={args.max_len}, "
          f"N={args.n_requests})", ["distribution", "mean len", "contiguous", "paged"], rows)

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(DISTS)); w = 0.35
    ax.bar(x - w / 2, [res[d]["contiguous"] * 100 for d in DISTS], w, label="contiguous", color=COLORS["contiguous"])
    ax.bar(x + w / 2, [res[d]["paged"] * 100 for d in DISTS], w, label="paged", color=COLORS["paged"])
    for i, d in enumerate(DISTS):
        ax.text(i - w / 2, res[d]["contiguous"] * 100 + 1, f"{res[d]['contiguous']:.0%}", ha="center", fontsize=9)
        ax.text(i + w / 2, res[d]["paged"] * 100 + 1, f"{res[d]['paged']:.0%}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(DISTS); ax.set_ylim(0, 110)
    ax.set_ylabel("KV slot utilization (%)"); ax.set_title("Used / reserved KV slots")
    ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(out, "fig1_utilization.png"), dpi=150); plt.close(fig)
    return res


# ----------------------------------------------------------------------------
# Experiment 2: waste vs length
# ----------------------------------------------------------------------------
def exp2_waste(args, out):
    lengths = np.arange(1, args.max_len + 1)
    block_sizes = [4, 16, 64]
    waste_paged = {B: [blocks_used_via_cache(int(L), B, args.max_len) * B - int(L) for L in lengths]
                   for B in block_sizes}
    waste_contig = args.max_len - lengths

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(lengths, waste_contig, color=COLORS["contiguous"], lw=2, label=f"contiguous (max_len={args.max_len})")
    for B, ls in zip(block_sizes, ["-", "--", ":"]):
        ax.plot(lengths, waste_paged[B], color=COLORS["paged"], ls=ls, lw=1.5, label=f"paged, block_size={B}")
    ax.set_xlabel("sequence length (tokens)"); ax.set_ylabel("wasted slots")
    ax.set_title("Wasted KV slots per sequence"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig2_waste_vs_length.png"), dpi=150); plt.close(fig)

    rows = [[B, max(waste_paged[B]), f"{np.mean(waste_paged[B]):.2f}", B - 1] for B in block_sizes]
    table("[2] Waste per sequence, paged (contiguous waste = max_len - len)",
          ["block_size", "max waste", "mean waste", "bound B-1"], rows)
    return {str(B): {"max": int(max(w)), "mean": float(np.mean(w))} for B, w in waste_paged.items()}


# ----------------------------------------------------------------------------
# Experiment 3: max concurrent sequences under a fixed budget
# ----------------------------------------------------------------------------
def exp3_capacity(args, rng, out):
    budgets = [512, 1024, 2048, 4096, 8192, 16384]      # in token slots
    res, rows = {}, []
    fig, axes = plt.subplots(1, len(DISTS), figsize=(5 * len(DISTS), 4), sharey=False)
    for ax, d in zip(axes, DISTS):
        means, stds, contig = [], [], []
        for b in budgets:
            counts = []
            for _ in range(args.trials):
                stream = sample_lengths(d, 4 * (b // args.min_len) + 10, rng, args.min_len, args.max_len)
                counts.append(max_concurrent_paged(stream, b, args.block_size))
            means.append(np.mean(counts)); stds.append(np.std(counts))
            contig.append(max_concurrent_contiguous(b, args.max_len))
            gain = means[-1] / max(contig[-1], 1)
            rows.append([d, b, f"{mb(b):.1f}", contig[-1], f"{means[-1]:.1f} ± {stds[-1]:.1f}", f"{gain:.2f}x"])
        res[d] = {"budgets": budgets, "contiguous": contig, "paged_mean": means, "paged_std": stds}
        means, stds = np.array(means), np.array(stds)
        ax.plot(budgets, contig, "o-", color=COLORS["contiguous"], label="contiguous")
        ax.plot(budgets, means, "o-", color=COLORS["paged"], label=f"paged (B={args.block_size})")
        ax.fill_between(budgets, means - stds, means + stds, color=COLORS["paged"], alpha=0.2)
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xlabel("KV budget (token slots)"); ax.set_ylabel("max concurrent sequences")
        ax.set_title(d); ax.grid(alpha=0.3, which="both"); ax.legend()
    fig.suptitle("Capacity under a fixed KV budget"); fig.tight_layout()
    fig.savefig(os.path.join(out, "fig3_max_concurrent.png"), dpi=150); plt.close(fig)
    table(f"[3] Max concurrent sequences (bytes/token={BYTES_PER_TOKEN}, trials={args.trials})",
          ["dist", "budget slots", "budget MB", "contiguous", "paged", "gain"], rows)
    return res


# ----------------------------------------------------------------------------
# Experiment 4: block size sweep
# ----------------------------------------------------------------------------
def exp4_block_sweep(args, rng, out):
    block_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    block_sizes = [B for B in block_sizes if B <= args.max_len]
    budget = args.sweep_budget
    res = {d: {"util": [], "capacity": [], "table_len": []} for d in ("lognormal", "uniform")}
    rows = []
    for d in res:
        lens = sample_lengths(d, args.n_requests, rng, args.min_len, args.max_len)
        for B in block_sizes:
            _, p, tl = utilization(lens, B, args.max_len)
            caps = [max_concurrent_paged(
                        sample_lengths(d, 4 * (budget // args.min_len) + 10, rng, args.min_len, args.max_len),
                        budget, B) for _ in range(args.trials)]
            res[d]["util"].append(p); res[d]["capacity"].append(float(np.mean(caps))); res[d]["table_len"].append(tl)
            rows.append([d, B, f"{p:.1%}", f"{np.mean(caps):.1f}", f"{tl:.1f}"])
    contig_cap = max_concurrent_contiguous(budget, args.max_len)
    table(f"[4] Block size sweep (budget={budget} slots; contiguous capacity={contig_cap})",
          ["dist", "block_size", "paged util", "max concurrent", "mean table len"], rows)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for d, ls in zip(res, ["-", "--"]):
        axes[0].plot(block_sizes, np.array(res[d]["util"]) * 100, "o" + ls, label=d)
        axes[1].plot(block_sizes, res[d]["capacity"], "o" + ls, label=d)
        axes[2].plot(block_sizes, res[d]["table_len"], "o" + ls, label=d)
    axes[1].axhline(contig_cap, color=COLORS["contiguous"], ls=":", label="contiguous")
    for ax, t, yl in zip(axes, ["Utilization vs block size", "Capacity vs block size", "Block-table length vs block size"],
                         ["utilization (%)", "max concurrent sequences", "mean blocks per sequence"]):
        ax.set_xscale("log", base=2); ax.set_xlabel("block size (tokens)"); ax.set_ylabel(yl)
        ax.set_title(t); ax.grid(alpha=0.3); ax.legend()
    axes[2].set_yscale("log")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig4_block_size_sweep.png"), dpi=150); plt.close(fig)
    res["block_sizes"] = block_sizes; res["contiguous_capacity"] = contig_cap
    return res


# ----------------------------------------------------------------------------
# Experiment 5: timings
# ----------------------------------------------------------------------------
def exp5_timings(args, out):
    block_sizes = [B for B in [1, 4, 16, 64] if B <= args.max_len]
    L = args.max_len
    reserve_ns, gather_us = [], []
    ref = torch.randn(L, N_HEADS, HEAD_DIM)

    for B in block_sizes:
        # reserve(): cost per token, averaged over repeated fills
        cache = make_cache(math.ceil(L / B) + 1, B)
        reps, t0 = 50, time.perf_counter()
        for r in range(reps):
            fill(cache, r, L); cache.free(r)
        reserve_ns.append((time.perf_counter() - t0) / (reps * L) * 1e9)

        # gather(): realistic tensor sizes, one layer, one full-length sequence
        cache = PagedKVCache(1, math.ceil(L / B) + 1, B, N_HEADS, HEAD_DIM)
        cache.add_sequence(0)
        for i in range(L):
            phys, slot = cache.reserve(0)
            cache.write(0, phys, slot, ref[i], ref[i])
        for _ in range(20): cache.gather(0, 0)           # warmup
        reps, t0 = 300, time.perf_counter()
        for _ in range(reps): cache.gather(0, 0)
        gather_us.append((time.perf_counter() - t0) / reps * 1e6)
        k, _v = cache.gather(0, 0)
        assert torch.equal(k, ref), "gather mismatch"

    reps, t0 = 300, time.perf_counter()
    for _ in range(reps):
        ref[:L].clone(); ref[:L].clone()      # K and V, to match gather()
    contig_us = (time.perf_counter() - t0) / reps * 1e6

    rows = [[B, f"{r:.0f}", f"{g:.1f}"] for B, r, g in zip(block_sizes, reserve_ns, gather_us)]
    rows.append(["contiguous slice+copy", "-", f"{contig_us:.1f}"])
    table(f"[5] Timings (CPU, seq_len={L}, one layer; gather includes K and V)",
          ["block_size", "reserve ns/token", "gather µs/call"], rows)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar([str(b) for b in block_sizes], reserve_ns, color=COLORS["paged"])
    axes[0].set_xlabel("block size"); axes[0].set_ylabel("ns per token"); axes[0].set_title("reserve() cost")
    axes[1].bar([str(b) for b in block_sizes], gather_us, color=COLORS["paged"], label="paged gather")
    axes[1].axhline(contig_us, color=COLORS["contiguous"], ls="--", label="contiguous copy (K and V)")
    axes[1].set_xlabel("block size"); axes[1].set_ylabel("µs per call")
    axes[1].set_title(f"Read path at seq_len={L}"); axes[1].legend()
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig5_timings.png"), dpi=150); plt.close(fig)
    return {"block_sizes": block_sizes, "reserve_ns_per_token": reserve_ns,
            "gather_us": gather_us, "contiguous_copy_us": contig_us}


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--min-len", type=int, default=8)
    ap.add_argument("--block-size", type=int, default=4, help="block size for experiments 1-3")
    ap.add_argument("--n-requests", type=int, default=1000)
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--sweep-budget", type=int, default=4096, help="KV budget (slots) for experiment 4")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="bench_out")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    t0 = time.perf_counter()

    results = {"config": vars(args), "bytes_per_token": BYTES_PER_TOKEN}
    results["exp1_utilization"] = exp1_utilization(args, rng, args.out)
    results["exp2_waste"] = exp2_waste(args, args.out)
    results["exp3_capacity"] = exp3_capacity(args, rng, args.out)
    results["exp4_block_sweep"] = exp4_block_sweep(args, rng, args.out)
    results["exp5_timings"] = exp5_timings(args, args.out)

    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nDone in {time.perf_counter() - t0:.1f}s. Figures and results.json are in ./{args.out}/")


if __name__ == "__main__":
    main()
