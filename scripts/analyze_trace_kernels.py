#!/usr/bin/env python3
"""Detailed kernel-level analysis for a single stage trace.

Reports per-kernel timing breakdown, occupancy distribution, SM saturation,
and identifies the heaviest kernels that dominate execution time.

Usage:
    python scripts/analyze_trace_kernels.py /mnt/local_storage/mps_traces/no_mps/stage_attn/trace.json
    python scripts/analyze_trace_kernels.py /mnt/local_storage/mps_traces/mps/stage_moe/trace.json --top 20
    python scripts/analyze_trace_kernels.py trace.json --num-sms 132  # H100
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass
class KernelStats:
    name: str
    count: int = 0
    total_dur_us: float = 0.0
    min_dur_us: float = float("inf")
    max_dur_us: float = 0.0
    total_blocks: int = 0
    avg_occupancy: float = 0.0
    occupancy_sum: float = 0.0
    avg_warps_per_sm: float = 0.0
    warps_sum: float = 0.0
    avg_registers: float = 0.0
    avg_shared_mem: float = 0.0


def shorten_name(name: str) -> str:
    """Shorten kernel name for display."""
    # Strip "void " prefix
    if name.startswith("void "):
        name = name[5:]
    # Truncate at first template parameter
    if "<" in name:
        depth = 0
        for i, c in enumerate(name):
            if c == "<":
                depth += 1
                if depth == 1:
                    return name[:i] + "<...>"
            elif c == ">":
                depth -= 1
    return name[:100]


def classify_kernel(name: str) -> str:
    """Classify kernel into a high-level category."""
    nl = name.lower()
    if "gemm" in nl or "cutlass" in nl or "cublaslt" in nl or "cublas" in nl:
        return "GEMM"
    if "flash" in nl or "fmha" in nl or "attention" in nl:
        return "Attention"
    if "softmax" in nl:
        return "Softmax"
    if "layer_norm" in nl or "layernorm" in nl or "rms_norm" in nl:
        return "LayerNorm"
    if "elementwise" in nl or "vectorized" in nl:
        return "Elementwise"
    if "reduce" in nl or "welford" in nl:
        return "Reduction"
    if "copy" in nl or "memcpy" in nl:
        return "Copy"
    if "memset" in nl:
        return "Memset"
    if "scatter" in nl or "gather" in nl or "index" in nl:
        return "Index/Gather"
    if "topk" in nl or "sort" in nl:
        return "TopK/Sort"
    return "Other"


def main():
    parser = argparse.ArgumentParser(
        description="Detailed kernel analysis for a single stage trace."
    )
    parser.add_argument("trace_path", type=Path, help="Path to trace.json")
    parser.add_argument("--top", type=int, default=15,
                        help="Number of top kernels to show (default: 15)")
    parser.add_argument("--skip-warmup", type=int, default=5,
                        help="Skip first N annotation groups (default: 5)")
    parser.add_argument("--category", action="store_true",
                        help="Group kernels by category (GEMM, Attention, etc.)")
    parser.add_argument("--num-sms", type=int, default=132,
                        help="Number of SMs on the GPU (default: 132 for H100)")
    args = parser.parse_args()

    if not args.trace_path.exists():
        print(f"Error: {args.trace_path} not found")
        sys.exit(1)

    print(f"Loading {args.trace_path} ...")
    with open(args.trace_path) as f:
        trace = json.load(f)

    events = trace.get("traceEvents", [])

    # Extract annotations to find iteration boundaries
    annotations = []
    for e in events:
        if isinstance(e, dict) and e.get("cat") == "user_annotation" and e.get("dur") is not None:
            annotations.append(e)
    annotations.sort(key=lambda e: e["ts"])

    # Find forward annotations to determine iteration boundaries
    forward_annots = [a for a in annotations if ".forward" in a.get("name", "")]
    backward_annots = [a for a in annotations if ".backward" in a.get("name", "")]

    if forward_annots and len(forward_annots) > args.skip_warmup:
        # Use annotation after warmup to set time filter
        first_measured = forward_annots[args.skip_warmup]
        time_start = first_measured["ts"]
        # End at last backward annotation
        if backward_annots:
            last_bwd = backward_annots[-1]
            time_end = last_bwd["ts"] + last_bwd.get("dur", 0)
        else:
            time_end = float("inf")
        print(f"Skipping {args.skip_warmup} warmup iterations, analyzing from iter {args.skip_warmup}")
        n_measured = len(forward_annots) - args.skip_warmup
        print(f"Measured iterations: {n_measured}")
    else:
        time_start = 0
        time_end = float("inf")
        n_measured = len(forward_annots) or 1

    # Extract kernels in measurement window
    kernels = []
    for e in events:
        if not isinstance(e, dict) or e.get("cat") != "kernel":
            continue
        if e["ts"] < time_start or e["ts"] > time_end:
            continue
        kernels.append(e)

    if not kernels:
        print("No kernels found in measurement window")
        sys.exit(0)

    print(f"Kernels in measurement window: {len(kernels)}")

    # Aggregate by kernel name
    stats: dict[str, KernelStats] = {}
    for k in kernels:
        name = k.get("name", "unknown")
        a = k.get("args", {})
        dur = k.get("dur", 0)

        if name not in stats:
            stats[name] = KernelStats(name=name)
        s = stats[name]
        s.count += 1
        s.total_dur_us += dur
        s.min_dur_us = min(s.min_dur_us, dur)
        s.max_dur_us = max(s.max_dur_us, dur)
        s.occupancy_sum += a.get("est. achieved occupancy %", 0)
        s.warps_sum += a.get("warps per SM", 0)
        grid = a.get("grid", [0, 0, 0])
        block = a.get("block", [0, 0, 0])
        s.total_blocks += grid[0] * grid[1] * grid[2]
        s.avg_registers = a.get("registers per thread", 0)
        s.avg_shared_mem = a.get("shared memory", 0)

    # Finalize averages
    for s in stats.values():
        if s.count > 0:
            s.avg_occupancy = s.occupancy_sum / s.count
            s.avg_warps_per_sm = s.warps_sum / s.count

    total_kernel_time = sum(s.total_dur_us for s in stats.values())

    # Sort by total time
    ranked = sorted(stats.values(), key=lambda s: -s.total_dur_us)

    # Print top kernels
    print(f"\n{'='*100}")
    print(f"  Top {args.top} kernels by total time (total: {total_kernel_time/1000:.1f} ms, "
          f"{total_kernel_time/1000/n_measured:.1f} ms/iter)")
    print(f"{'='*100}")
    print(f"  {'%':>5}  {'Total(ms)':>9}  {'Count':>6}  {'Avg(us)':>8}  {'Occ%':>5}  "
          f"{'Warps/SM':>8}  {'Cat':>10}  {'Kernel'}")
    print(f"  {'-'*5}  {'-'*9}  {'-'*6}  {'-'*8}  {'-'*5}  {'-'*8}  {'-'*10}  {'-'*40}")

    cumulative = 0.0
    for s in ranked[:args.top]:
        pct = s.total_dur_us / total_kernel_time * 100 if total_kernel_time > 0 else 0
        cumulative += pct
        cat = classify_kernel(s.name)
        print(f"  {pct:5.1f}  {s.total_dur_us/1000:9.2f}  {s.count:6d}  "
              f"{s.total_dur_us/s.count:8.1f}  {s.avg_occupancy:5.1f}  "
              f"{s.avg_warps_per_sm:8.1f}  {cat:>10}  {shorten_name(s.name)}")

    remaining = 100.0 - cumulative
    print(f"  {remaining:5.1f}  {'':>9}  {'':>6}  {'':>8}  {'':>5}  {'':>8}  {'':>10}  (remaining {len(ranked) - args.top} kernels)")

    # Category breakdown
    if args.category:
        cat_time: dict[str, float] = defaultdict(float)
        cat_count: dict[str, int] = defaultdict(int)
        cat_occ: dict[str, float] = defaultdict(float)
        cat_occ_weight: dict[str, float] = defaultdict(float)

        for s in stats.values():
            cat = classify_kernel(s.name)
            cat_time[cat] += s.total_dur_us
            cat_count[cat] += s.count
            cat_occ[cat] += s.occupancy_sum
            cat_occ_weight[cat] += s.total_dur_us

        print(f"\n{'='*70}")
        print(f"  Kernel category breakdown")
        print(f"{'='*70}")
        print(f"  {'Category':>12}  {'%':>5}  {'Total(ms)':>9}  {'Count':>6}  {'Wtd Occ%':>8}")
        print(f"  {'-'*12}  {'-'*5}  {'-'*9}  {'-'*6}  {'-'*8}")

        for cat, total in sorted(cat_time.items(), key=lambda x: -x[1]):
            pct = total / total_kernel_time * 100 if total_kernel_time > 0 else 0
            # Duration-weighted occupancy
            wtd_occ = 0
            for s in stats.values():
                if classify_kernel(s.name) == cat:
                    wtd_occ += s.occupancy_sum * (s.total_dur_us / total) if total > 0 else 0
            wtd_occ = wtd_occ / cat_count[cat] * (cat_count[cat] / max(1, sum(1 for s in stats.values() if classify_kernel(s.name) == cat)))
            # Simpler: just use straight average
            n_unique = sum(1 for s in stats.values() if classify_kernel(s.name) == cat)
            avg_occ = cat_occ[cat] / cat_count[cat] if cat_count[cat] > 0 else 0

            print(f"  {cat:>12}  {pct:5.1f}  {total/1000:9.2f}  {cat_count[cat]:6d}  {avg_occ:8.1f}")

    # Occupancy distribution
    all_occupancies = []
    for k in kernels:
        occ = k.get("args", {}).get("est. achieved occupancy %", 0)
        if occ > 0:
            all_occupancies.append(occ)

    if all_occupancies:
        print(f"\n  Occupancy distribution ({len(all_occupancies)} kernels):")
        buckets = [(0, 25), (25, 50), (50, 75), (75, 90), (90, 100), (100, 101)]
        labels = ["0-25%", "25-50%", "50-75%", "75-90%", "90-100%", "100%"]
        for (lo, hi), label in zip(buckets, labels):
            count = sum(1 for o in all_occupancies if lo <= o < hi)
            bar = "#" * (count * 40 // len(all_occupancies)) if all_occupancies else ""
            print(f"    {label:>7}: {count:5d} ({count/len(all_occupancies)*100:5.1f}%) {bar}")

    # GPU utilization (kernel active time vs wall time)
    if kernels:
        k_start = min(k["ts"] for k in kernels)
        k_end = max(k["ts"] + k.get("dur", 0) for k in kernels)
        wall = k_end - k_start
        if wall > 0:
            util = total_kernel_time / wall * 100
            print(f"\n  GPU kernel utilization: {util:.1f}% (kernel time / wall time)")
            print(f"    Wall time: {wall/1000:.1f} ms, Kernel time: {total_kernel_time/1000:.1f} ms")
            print(f"    Gap time (idle): {(wall - total_kernel_time)/1000:.1f} ms")

    # SM saturation analysis
    # Determines what fraction of kernel time saturates all SMs, leaving no
    # headroom for concurrent kernels from another MPS client.
    num_sms = args.num_sms
    print(f"\n{'='*70}")
    print(f"  SM saturation analysis (GPU has {num_sms} SMs)")
    print(f"{'='*70}")

    saturating_time_us = 0.0  # time from kernels that fill all SMs
    partial_time_us = 0.0     # time from kernels that leave SM headroom
    light_time_us = 0.0       # time from kernels using <25% of SMs

    sm_usage_records = []  # (blocks_per_sm, occupancy, duration, name) for each kernel
    for k in kernels:
        a = k.get("args", {})
        dur = k.get("dur", 0)
        grid = a.get("grid", [0, 0, 0])
        total_blocks = grid[0] * grid[1] * grid[2]
        blocks_per_sm = a.get("blocks per SM", 0)
        occ = a.get("est. achieved occupancy %", 0)

        # A kernel saturates SMs if it launches >= num_sms blocks
        # (each SM gets at least one block)
        if total_blocks >= num_sms:
            saturating_time_us += dur
        elif total_blocks >= num_sms * 0.25:
            partial_time_us += dur
        else:
            light_time_us += dur

        if dur > 0:
            sm_usage_records.append((blocks_per_sm, occ, dur, k.get("name", "")))

    if total_kernel_time > 0:
        sat_pct = saturating_time_us / total_kernel_time * 100
        partial_pct = partial_time_us / total_kernel_time * 100
        light_pct = light_time_us / total_kernel_time * 100

        print(f"  SM saturation by kernel time:")
        print(f"    Saturating (>={num_sms} blocks, fills all SMs): "
              f"{saturating_time_us/1000:8.2f} ms ({sat_pct:5.1f}%)")
        print(f"    Partial    (25-99% of SMs):                     "
              f"{partial_time_us/1000:8.2f} ms ({partial_pct:5.1f}%)")
        print(f"    Light      (<25% of SMs):                       "
              f"{light_time_us/1000:8.2f} ms ({light_pct:5.1f}%)")

        print(f"\n  MPS overlap headroom:")
        if sat_pct > 80:
            print(f"    -> {sat_pct:.0f}% of kernel time saturates all SMs.")
            print(f"    -> Almost no headroom for concurrent kernels from another stage.")
            print(f"    -> MPS CANNOT help — hardware has no free execution slots.")
        elif sat_pct > 50:
            print(f"    -> {sat_pct:.0f}% of kernel time saturates all SMs.")
            print(f"    -> Limited headroom ({100-sat_pct:.0f}% of time has free SMs).")
            print(f"    -> MPS can help during non-saturating windows only.")
        else:
            print(f"    -> Only {sat_pct:.0f}% of kernel time saturates all SMs.")
            print(f"    -> Good headroom ({100-sat_pct:.0f}% of time has free SMs).")
            print(f"    -> MPS should enable concurrent execution from another stage.")

    # Time-weighted blocks-per-SM distribution
    if sm_usage_records:
        weighted_bpsm = sum(bpsm * dur for bpsm, _, dur, _ in sm_usage_records) / total_kernel_time
        weighted_occ = sum(occ * dur for _, occ, dur, _ in sm_usage_records) / total_kernel_time

        print(f"\n  Time-weighted averages:")
        print(f"    Blocks per SM:     {weighted_bpsm:.1f}")
        print(f"    Occupancy:         {weighted_occ:.1f}%")

        # Distribution of blocks-per-SM
        print(f"\n  Blocks-per-SM distribution (by kernel time):")
        bpsm_buckets = [(0, 1), (1, 10), (10, 50), (50, 100), (100, float("inf"))]
        bpsm_labels = ["<1", "1-10", "10-50", "50-100", "100+"]
        for (lo, hi), label in zip(bpsm_buckets, bpsm_labels):
            bucket_time = sum(dur for bpsm, _, dur, _ in sm_usage_records if lo <= bpsm < hi)
            pct = bucket_time / total_kernel_time * 100 if total_kernel_time > 0 else 0
            bar = "#" * int(pct / 2.5)
            print(f"    {label:>7}: {bucket_time/1000:8.2f} ms ({pct:5.1f}%) {bar}")


if __name__ == "__main__":
    main()
