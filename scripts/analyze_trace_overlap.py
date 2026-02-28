#!/usr/bin/env python3
"""Analyze PyTorch Profiler traces for kernel overlap between pipeline stages.

Loads Chrome trace JSON files from two pipeline stages (e.g., attention and MoE)
and determines whether their CUDA kernels overlap on the GPU timeline. Compares
no_mps vs mps variants to quantify MPS overlap benefit.

Usage:
    python scripts/analyze_trace_overlap.py /mnt/local_storage/mps_traces
    python scripts/analyze_trace_overlap.py /mnt/local_storage/mps_traces --variant mps
    python scripts/analyze_trace_overlap.py /mnt/local_storage/mps_traces --stages attn moe

The trace directory should contain:
    {variant}/stage_{stage_a}/trace.json
    {variant}/stage_{stage_b}/trace.json

For each variant (no_mps, mps).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class KernelEvent:
    name: str
    ts: float  # start time in us
    dur: float  # duration in us
    stream: int
    occupancy: float  # est. achieved occupancy %
    blocks_per_sm: float
    warps_per_sm: float
    grid: list[int]
    block: list[int]
    registers: int
    shared_mem: int
    stage: str = ""

    @property
    def end(self) -> float:
        return self.ts + self.dur


@dataclass
class Annotation:
    name: str
    ts: float
    dur: float

    @property
    def end(self) -> float:
        return self.ts + self.dur


@dataclass
class OverlapResult:
    """Result of overlap analysis between two stages."""

    variant: str
    stage_a: str
    stage_b: str
    # Per-iteration results
    iterations: list[IterationOverlap] = field(default_factory=list)
    # Aggregate
    total_a_kernel_time_us: float = 0.0
    total_b_kernel_time_us: float = 0.0
    total_overlap_us: float = 0.0
    total_gap_us: float = 0.0


@dataclass
class IterationOverlap:
    iteration: int
    a_kernel_time_us: float
    b_kernel_time_us: float
    overlap_us: float
    a_kernel_count: int
    b_kernel_count: int
    wall_time_us: float  # wall time from first kernel to last kernel


def load_trace(path: Path) -> dict:
    """Load a Chrome trace JSON file."""
    with open(path) as f:
        return json.load(f)


def extract_kernels(trace: dict, stage: str = "") -> list[KernelEvent]:
    """Extract CUDA kernel events from a trace."""
    events = trace.get("traceEvents", [])
    kernels = []
    for e in events:
        if not isinstance(e, dict) or e.get("cat") != "kernel":
            continue
        args = e.get("args", {})
        kernels.append(
            KernelEvent(
                name=e.get("name", ""),
                ts=e["ts"],
                dur=e.get("dur", 0),
                stream=args.get("stream", -1),
                occupancy=args.get("est. achieved occupancy %", 0),
                blocks_per_sm=args.get("blocks per SM", 0),
                warps_per_sm=args.get("warps per SM", 0),
                grid=args.get("grid", [0, 0, 0]),
                block=args.get("block", [0, 0, 0]),
                registers=args.get("registers per thread", 0),
                shared_mem=args.get("shared memory", 0),
                stage=stage,
            )
        )
    kernels.sort(key=lambda k: k.ts)
    return kernels


def extract_annotations(trace: dict) -> list[Annotation]:
    """Extract user_annotation events (forward/backward labels)."""
    events = trace.get("traceEvents", [])
    annots = []
    for e in events:
        if not isinstance(e, dict) or e.get("cat") != "user_annotation":
            continue
        if e.get("dur") is None:
            continue
        annots.append(
            Annotation(
                name=e.get("name", ""),
                ts=e["ts"],
                dur=e["dur"],
            )
        )
    annots.sort(key=lambda a: a.ts)
    return annots


def compute_interval_overlap(
    intervals_a: list[tuple[float, float]],
    intervals_b: list[tuple[float, float]],
) -> float:
    """Compute total overlap duration between two sets of [start, end) intervals.

    Uses a sweep-line algorithm. Both lists must be sorted by start time.
    """
    if not intervals_a or not intervals_b:
        return 0.0

    # Merge into event list: +1 for start, -1 for end
    events: list[tuple[float, int, str]] = []
    for s, e in intervals_a:
        events.append((s, 1, "a"))
        events.append((e, -1, "a"))
    for s, e in intervals_b:
        events.append((s, 1, "b"))
        events.append((e, -1, "b"))
    events.sort(key=lambda x: (x[0], x[1]))

    active_a = 0
    active_b = 0
    overlap = 0.0
    prev_t = events[0][0]

    for t, delta, src in events:
        if active_a > 0 and active_b > 0:
            overlap += t - prev_t
        prev_t = t
        if src == "a":
            active_a += delta
        else:
            active_b += delta

    return overlap


def compute_total_active_time(intervals: list[tuple[float, float]]) -> float:
    """Compute total active time from a set of intervals (merging overlapping ones)."""
    if not intervals:
        return 0.0
    merged = []
    for s, e in sorted(intervals):
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return sum(e - s for s, e in merged)


def segment_by_iteration(
    annotations: list[Annotation],
    kernels: list[KernelEvent],
    stage_prefix: str,
) -> list[list[KernelEvent]]:
    """Group kernels into iterations based on forward/backward annotation pairs.

    Each iteration = one forward + one backward annotation span.
    Kernels falling within those time ranges are assigned to that iteration.
    """
    # Pair up forward/backward annotations
    forwards = [a for a in annotations if a.name == f"{stage_prefix}.forward"]
    backwards = [a for a in annotations if a.name == f"{stage_prefix}.backward"]

    if not forwards or not backwards:
        # Fall back: treat all kernels as one iteration
        return [kernels]

    iterations = []
    for i, (fwd, bwd) in enumerate(zip(forwards, backwards)):
        iter_start = fwd.ts
        iter_end = bwd.end
        iter_kernels = [k for k in kernels if k.ts >= iter_start and k.end <= iter_end]
        iterations.append(iter_kernels)

    return iterations


def analyze_variant(
    trace_dir: Path,
    variant: str,
    stage_a: str,
    stage_b: str,
    skip_warmup: int = 5,
) -> OverlapResult:
    """Analyze kernel overlap for a single variant."""
    path_a = trace_dir / variant / f"stage_{stage_a}" / "trace.json"
    path_b = trace_dir / variant / f"stage_{stage_b}" / "trace.json"

    if not path_a.exists() or not path_b.exists():
        print(f"  Trace files not found for variant '{variant}':")
        if not path_a.exists():
            print(f"    Missing: {path_a}")
        if not path_b.exists():
            print(f"    Missing: {path_b}")
        return OverlapResult(variant=variant, stage_a=stage_a, stage_b=stage_b)

    print(f"  Loading {path_a} ...")
    trace_a = load_trace(path_a)
    print(f"  Loading {path_b} ...")
    trace_b = load_trace(path_b)

    kernels_a = extract_kernels(trace_a, stage_a)
    kernels_b = extract_kernels(trace_b, stage_b)
    annots_a = extract_annotations(trace_a)
    annots_b = extract_annotations(trace_b)

    print(f"  {stage_a}: {len(kernels_a)} kernels, {len(annots_a)} annotations")
    print(f"  {stage_b}: {len(kernels_b)} kernels, {len(annots_b)} annotations")

    # Segment into iterations
    iters_a = segment_by_iteration(annots_a, kernels_a, stage_a)
    iters_b = segment_by_iteration(annots_b, kernels_b, stage_b)

    print(f"  {stage_a}: {len(iters_a)} iterations detected")
    print(f"  {stage_b}: {len(iters_b)} iterations detected")

    # Skip warmup iterations
    iters_a = iters_a[skip_warmup:]
    iters_b = iters_b[skip_warmup:]
    n_iters = min(len(iters_a), len(iters_b))

    if n_iters == 0:
        print(f"  No iterations remaining after skipping {skip_warmup} warmup")
        return OverlapResult(variant=variant, stage_a=stage_a, stage_b=stage_b)

    print(f"  Analyzing {n_iters} iterations (after {skip_warmup} warmup) ...")

    result = OverlapResult(variant=variant, stage_a=stage_a, stage_b=stage_b)

    for i in range(n_iters):
        ka = iters_a[i]
        kb = iters_b[i]

        intervals_a = [(k.ts, k.end) for k in ka]
        intervals_b = [(k.ts, k.end) for k in kb]

        a_active = compute_total_active_time(intervals_a)
        b_active = compute_total_active_time(intervals_b)
        overlap = compute_interval_overlap(intervals_a, intervals_b)

        # Wall time: from first kernel start to last kernel end across both stages
        all_kernels = ka + kb
        if all_kernels:
            wall_start = min(k.ts for k in all_kernels)
            wall_end = max(k.end for k in all_kernels)
            wall_time = wall_end - wall_start
        else:
            wall_time = 0.0

        result.iterations.append(
            IterationOverlap(
                iteration=i,
                a_kernel_time_us=a_active,
                b_kernel_time_us=b_active,
                overlap_us=overlap,
                a_kernel_count=len(ka),
                b_kernel_count=len(kb),
                wall_time_us=wall_time,
            )
        )

        result.total_a_kernel_time_us += a_active
        result.total_b_kernel_time_us += b_active
        result.total_overlap_us += overlap

    return result


def print_occupancy_stats(trace_dir: Path, variant: str, stage: str) -> None:
    """Print occupancy and kernel statistics for a stage."""
    path = trace_dir / variant / f"stage_{stage}" / "trace.json"
    if not path.exists():
        return

    trace = load_trace(path)
    kernels = extract_kernels(trace, stage)

    if not kernels:
        print(f"  No kernels found for {stage}")
        return

    occupancies = [k.occupancy for k in kernels if k.occupancy > 0]
    durations = [k.dur for k in kernels]

    # Weight occupancy by duration (longer kernels matter more)
    if occupancies and durations:
        total_dur = sum(durations)
        weighted_occ = sum(k.occupancy * k.dur for k in kernels if k.occupancy > 0) / total_dur

        print(f"\n  {stage} kernel stats ({variant}):")
        print(f"    Kernels:           {len(kernels)}")
        print(f"    Total kernel time: {sum(durations)/1000:.1f} ms")
        print(f"    Mean duration:     {sum(durations)/len(durations):.1f} us")
        print(f"    Median duration:   {sorted(durations)[len(durations)//2]:.1f} us")
        print(f"    Max duration:      {max(durations):.1f} us")
        print(f"    Occupancy (weighted by dur): {weighted_occ:.1f}%")
        print(f"    Occupancy range:   {min(occupancies):.1f}% - {max(occupancies):.1f}%")

        # Top 5 kernel types by total time
        kernel_time: dict[str, float] = defaultdict(float)
        kernel_count: dict[str, int] = defaultdict(int)
        for k in kernels:
            short_name = _shorten_kernel_name(k.name)
            kernel_time[short_name] += k.dur
            kernel_count[short_name] += 1

        print(f"    Top kernels by total time:")
        for name, total in sorted(kernel_time.items(), key=lambda x: -x[1])[:10]:
            pct = total / sum(durations) * 100
            print(f"      {pct:5.1f}% ({total/1000:7.1f} ms, {kernel_count[name]:4d}x) {name}")


def _shorten_kernel_name(name: str) -> str:
    """Shorten a CUDA kernel name for display."""
    # Remove template parameters for readability
    # void at::native::vectorized_elementwise_kernel<4, ...> -> vectorized_elementwise_kernel
    if "::" in name:
        parts = name.split("::")
        # Find the function name (last part before template)
        for i, p in enumerate(parts):
            if "<" in p:
                base = p[: p.index("<")]
                # Include one level of namespace
                if i > 0:
                    return f"{parts[i-1]}::{base}"
                return base
        return parts[-1][:80]
    return name[:80]


def print_overlap_report(result: OverlapResult) -> None:
    """Print a formatted overlap analysis report."""
    if not result.iterations:
        print(f"\n{'='*70}")
        print(f"  {result.variant}: No data")
        return

    n = len(result.iterations)
    print(f"\n{'='*70}")
    print(f"  Variant: {result.variant}")
    print(f"  Stages:  {result.stage_a} vs {result.stage_b}")
    print(f"  Iterations: {n}")
    print(f"{'='*70}")

    # Per-iteration table
    print(
        f"\n  {'Iter':>4}  {'A kern(ms)':>10}  {'B kern(ms)':>10}  "
        f"{'Overlap(ms)':>11}  {'Ovlp%':>6}  {'Wall(ms)':>9}  {'Speedup':>7}"
    )
    print(f"  {'-'*4}  {'-'*10}  {'-'*10}  {'-'*11}  {'-'*6}  {'-'*9}  {'-'*7}")

    for it in result.iterations:
        a_ms = it.a_kernel_time_us / 1000
        b_ms = it.b_kernel_time_us / 1000
        ovlp_ms = it.overlap_us / 1000
        wall_ms = it.wall_time_us / 1000
        total_serial = a_ms + b_ms
        ovlp_pct = (it.overlap_us / total_serial * 100 * 1000) if total_serial > 0 else 0
        # Correct overlap percentage calculation
        ovlp_pct = (
            (it.overlap_us / (it.a_kernel_time_us + it.b_kernel_time_us) * 100)
            if (it.a_kernel_time_us + it.b_kernel_time_us) > 0
            else 0
        )
        speedup = total_serial / wall_ms if wall_ms > 0 else 0

        print(
            f"  {it.iteration:4d}  {a_ms:10.2f}  {b_ms:10.2f}  "
            f"{ovlp_ms:11.2f}  {ovlp_pct:5.1f}%  {wall_ms:9.2f}  {speedup:6.2f}x"
        )

    # Aggregates
    total_a = result.total_a_kernel_time_us / 1000
    total_b = result.total_b_kernel_time_us / 1000
    total_ovlp = result.total_overlap_us / 1000
    total_serial = total_a + total_b
    total_wall = sum(it.wall_time_us for it in result.iterations) / 1000
    avg_ovlp_pct = (
        (result.total_overlap_us / (result.total_a_kernel_time_us + result.total_b_kernel_time_us) * 100)
        if (result.total_a_kernel_time_us + result.total_b_kernel_time_us) > 0
        else 0
    )

    print(f"\n  Summary:")
    print(f"    {result.stage_a} total kernel time: {total_a:.2f} ms")
    print(f"    {result.stage_b} total kernel time: {total_b:.2f} ms")
    print(f"    Serial sum:   {total_serial:.2f} ms")
    print(f"    Wall time:    {total_wall:.2f} ms")
    print(f"    Overlap:      {total_ovlp:.2f} ms ({avg_ovlp_pct:.1f}%)")
    if total_wall > 0:
        print(f"    Speedup:      {total_serial / total_wall:.2f}x vs serial kernel sum")

    # Interpretation
    print(f"\n  Interpretation:")
    if avg_ovlp_pct < 1.0:
        print(f"    -> Kernels are effectively serialized. No concurrent execution detected.")
    elif avg_ovlp_pct < 10.0:
        print(f"    -> Minimal overlap ({avg_ovlp_pct:.1f}%). Likely just kernel launch jitter.")
    elif avg_ovlp_pct < 30.0:
        print(f"    -> Moderate overlap. Some concurrent execution happening.")
    else:
        print(f"    -> Significant overlap ({avg_ovlp_pct:.1f}%). MPS enabling real concurrency.")


def print_sm_saturation_report(trace_dir: Path, variant: str, stage_a: str, stage_b: str, num_sms: int = 132) -> None:
    """Analyze GPU resource saturation: can stage B's kernels fit alongside stage A's?

    If stage A's kernels saturate all SMs (launch >= num_sms blocks), the CUDA
    hardware scheduler cannot start stage B's kernels concurrently, regardless
    of MPS. This is the fundamental limit on MPS-based overlap.
    """
    path_a = trace_dir / variant / f"stage_{stage_a}" / "trace.json"
    path_b = trace_dir / variant / f"stage_{stage_b}" / "trace.json"
    if not path_a.exists() or not path_b.exists():
        return

    trace_a = load_trace(path_a)
    trace_b = load_trace(path_b)
    kernels_a = extract_kernels(trace_a, stage_a)
    kernels_b = extract_kernels(trace_b, stage_b)

    print(f"\n{'='*70}")
    print(f"  SM saturation analysis ({variant}, {num_sms} SMs)")
    print(f"{'='*70}")

    for stage_name, kernels in [(stage_a, kernels_a), (stage_b, kernels_b)]:
        if not kernels:
            continue
        total_time = sum(k.dur for k in kernels)
        if total_time == 0:
            continue

        # Classify by SM saturation
        sat_time = sum(k.dur for k in kernels if k.grid[0] * k.grid[1] * k.grid[2] >= num_sms)
        sat_pct = sat_time / total_time * 100

        # Time-weighted occupancy
        wtd_occ = sum(k.occupancy * k.dur for k in kernels) / total_time
        # Time-weighted blocks per SM
        wtd_bpsm = sum(k.blocks_per_sm * k.dur for k in kernels) / total_time

        print(f"\n  {stage_name}:")
        print(f"    Total kernel time:          {total_time/1000:.2f} ms")
        print(f"    Time saturating all SMs:    {sat_time/1000:.2f} ms ({sat_pct:.1f}%)")
        print(f"    Time-weighted occupancy:    {wtd_occ:.1f}%")
        print(f"    Time-weighted blocks/SM:    {wtd_bpsm:.1f}")

    # Cross-stage analysis: what fraction of stage A's kernel time leaves
    # room for stage B's kernels?
    if kernels_a and kernels_b:
        total_a = sum(k.dur for k in kernels_a)
        total_b = sum(k.dur for k in kernels_b)
        sat_a = sum(k.dur for k in kernels_a if k.grid[0] * k.grid[1] * k.grid[2] >= num_sms)
        sat_b = sum(k.dur for k in kernels_b if k.grid[0] * k.grid[1] * k.grid[2] >= num_sms)
        sat_a_pct = sat_a / total_a * 100 if total_a else 0
        sat_b_pct = sat_b / total_b * 100 if total_b else 0

        print(f"\n  Cross-stage overlap potential:")
        print(f"    {stage_a}: {sat_a_pct:.0f}% of kernel time saturates all SMs")
        print(f"    {stage_b}: {sat_b_pct:.0f}% of kernel time saturates all SMs")

        # Theoretical max overlap: only possible during non-saturating windows
        max_overlap_window_a = (1 - sat_a / total_a) * total_a if total_a else 0
        max_overlap_window_b = (1 - sat_b / total_b) * total_b if total_b else 0
        # The overlap window is limited by the smaller non-saturating window
        max_overlap = min(max_overlap_window_a, max_overlap_window_b)
        total_serial = total_a + total_b
        max_overlap_pct = max_overlap / total_serial * 100 if total_serial else 0

        print(f"    Max theoretical overlap:    {max_overlap/1000:.2f} ms " f"({max_overlap_pct:.1f}% of serial sum)")

        if sat_a_pct > 80 or sat_b_pct > 80:
            dominant = stage_a if sat_a_pct > sat_b_pct else stage_b
            print(f"    -> {dominant} saturates GPU — MPS cannot schedule concurrent kernels")
            print(f"    -> Overlap requires reducing {dominant}'s per-kernel parallelism or")
            print(f"       increasing the other stage's kernel size to fill idle windows")
        elif max_overlap_pct > 20:
            print(f"    -> Good overlap potential. Non-saturating windows exist in both stages.")
        else:
            print(f"    -> Limited overlap potential even with MPS.")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze kernel overlap between pipeline stages from PyTorch Profiler traces."
    )
    parser.add_argument("trace_dir", type=Path, help="Root trace directory")
    parser.add_argument(
        "--variants", nargs="+", default=["no_mps", "mps"], help="Variants to analyze (default: no_mps mps)"
    )
    parser.add_argument("--stages", nargs=2, default=["attn", "moe"], help="Stage names (default: attn moe)")
    parser.add_argument("--skip-warmup", type=int, default=5, help="Number of warmup iterations to skip (default: 5)")
    parser.add_argument("--occupancy", action="store_true", help="Print detailed occupancy and kernel stats")
    parser.add_argument("--num-sms", type=int, default=132, help="Number of SMs on the GPU (default: 132 for H100)")
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Write machine-readable overlap/SM-saturation metrics to JSON file",
    )
    args = parser.parse_args()

    if not args.trace_dir.exists():
        print(f"Error: trace directory does not exist: {args.trace_dir}")
        sys.exit(1)

    stage_a, stage_b = args.stages
    results = {}

    for variant in args.variants:
        print(f"\n--- Analyzing variant: {variant} ---")
        result = analyze_variant(args.trace_dir, variant, stage_a, stage_b, args.skip_warmup)
        results[variant] = result
        print_overlap_report(result)

        if args.occupancy:
            print_occupancy_stats(args.trace_dir, variant, stage_a)
            print_occupancy_stats(args.trace_dir, variant, stage_b)
            print_sm_saturation_report(args.trace_dir, variant, stage_a, stage_b, args.num_sms)

    # Write machine-readable JSON output if requested
    if args.json_output is not None:
        json_metrics = {}
        for variant, result in results.items():
            entry = {"variant": variant, "iterations_analyzed": len(result.iterations)}
            total_kernel = result.total_a_kernel_time_us + result.total_b_kernel_time_us
            entry["overlap_pct"] = result.total_overlap_us / total_kernel * 100 if total_kernel > 0 else 0.0
            # SM saturation requires loading traces (use cached results if available)
            entry["attn_full_sm_pct"] = 0.0
            entry["moe_full_sm_pct"] = 0.0
            if result.iterations:
                path_a = args.trace_dir / variant / f"stage_{stage_a}" / "trace.json"
                path_b = args.trace_dir / variant / f"stage_{stage_b}" / "trace.json"
                if path_a.exists() and path_b.exists():
                    trace_a = load_trace(path_a)
                    trace_b = load_trace(path_b)
                    kernels_a = extract_kernels(trace_a, stage_a)
                    kernels_b = extract_kernels(trace_b, stage_b)
                    num_sms = args.num_sms
                    total_a = sum(k.dur for k in kernels_a)
                    total_b = sum(k.dur for k in kernels_b)
                    if total_a > 0:
                        sat_a = sum(k.dur for k in kernels_a if k.grid[0] * k.grid[1] * k.grid[2] >= num_sms)
                        entry["attn_full_sm_pct"] = sat_a / total_a * 100
                    if total_b > 0:
                        sat_b = sum(k.dur for k in kernels_b if k.grid[0] * k.grid[1] * k.grid[2] >= num_sms)
                        entry["moe_full_sm_pct"] = sat_b / total_b * 100
            json_metrics[variant] = entry

        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_output, "w") as f:
            json.dump(json_metrics, f, indent=2)
        print(f"\n  JSON metrics written to {args.json_output}")

    # Cross-variant comparison
    if len(results) == 2 and all(r.iterations for r in results.values()):
        variants = list(results.keys())
        r0, r1 = results[variants[0]], results[variants[1]]
        print(f"\n{'='*70}")
        print(f"  Cross-variant comparison: {variants[0]} vs {variants[1]}")
        print(f"{'='*70}")

        ovlp0 = (
            r0.total_overlap_us / (r0.total_a_kernel_time_us + r0.total_b_kernel_time_us) * 100
            if (r0.total_a_kernel_time_us + r0.total_b_kernel_time_us) > 0
            else 0
        )
        ovlp1 = (
            r1.total_overlap_us / (r1.total_a_kernel_time_us + r1.total_b_kernel_time_us) * 100
            if (r1.total_a_kernel_time_us + r1.total_b_kernel_time_us) > 0
            else 0
        )

        wall0 = sum(it.wall_time_us for it in r0.iterations) / 1000
        wall1 = sum(it.wall_time_us for it in r1.iterations) / 1000

        print(f"  {variants[0]:>10}: {ovlp0:.1f}% overlap, {wall0:.1f} ms total wall time")
        print(f"  {variants[1]:>10}: {ovlp1:.1f}% overlap, {wall1:.1f} ms total wall time")

        if ovlp1 > ovlp0 + 1.0:
            print(f"  -> {variants[1]} shows more overlap than {variants[0]} (+{ovlp1-ovlp0:.1f}pp)")
        elif ovlp0 > ovlp1 + 1.0:
            print(f"  -> {variants[0]} shows more overlap than {variants[1]} (+{ovlp0-ovlp1:.1f}pp)")
        else:
            print(f"  -> Similar overlap levels. MPS not enabling additional concurrency.")


if __name__ == "__main__":
    main()
