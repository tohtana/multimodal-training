"""Same-process overlap repro using two CUDA streams."""

from __future__ import annotations

import argparse
import time
from typing import Any

import torch
import torch.nn as nn

from examples.attn_moe_overlap.model_utils import (
    Qwen3AttentionStage,
    Qwen3MoEStageWithHead,
    create_qwen3_config,
)
from examples.attn_moe_overlap.repro_overlap_common import (
    build_case_id,
    build_case_payload,
    ensure_dirs,
    first_nonfinite_in_tensors,
    grad_checksums,
    load_baseline_artifact,
    make_error_payload,
    preflight_checks,
    save_baseline_artifact,
    seed_everything,
    tensor_numeric_diff,
    tolerance_for_dtype,
    write_case_json,
    dtype_from_name,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Same-process overlap NaN minimal repro")
    parser.add_argument("--mode", choices=["serial", "overlap"], required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--warmup-iters", type=int, default=3)
    parser.add_argument("--timed-iters", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--min-host-overlap-ms", type=float, default=0.01)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--json-output", type=str, default=None)
    parser.add_argument("--baseline-json", type=str, default=None)
    parser.add_argument("--attn-sample-elems", type=int, default=4096)
    parser.add_argument("--strict-schema", dest="strict_schema", action="store_true", default=True)
    parser.add_argument("--no-strict-schema", dest="strict_schema", action="store_false")
    return parser.parse_args()


def _baseline_metadata(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "experiment": "same_process",
        "dtype": args.dtype,
        "seq_len": int(args.seq_len),
        "seed": int(args.seed),
        "gpu_id": int(args.gpu_id),
        "batch_size": int(args.batch_size),
        "num_classes": int(args.num_classes),
        "attn_sample_elems": int(args.attn_sample_elems),
    }


def _metadata_match(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    for key, value in expected.items():
        if actual.get(key) != value:
            return False
    return True


def _collect_grad_tensors(attn_stage: nn.Module, moe_stage: nn.Module) -> list[tuple[str, torch.Tensor | None]]:
    tensors: list[tuple[str, torch.Tensor | None]] = []
    for name, param in attn_stage.named_parameters():
        tensors.append((f"attn.{name}.grad", param.grad))
        if len(tensors) >= 4:
            break
    for name, param in moe_stage.named_parameters():
        tensors.append((f"moe.{name}.grad", param.grad))
        if len(tensors) >= 8:
            break
    return tensors


def _build_models_and_inputs(args: argparse.Namespace) -> tuple[nn.Module, nn.Module, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch_dtype = dtype_from_name(args.dtype)
    device = torch.device(f"cuda:{args.gpu_id}")
    torch.cuda.set_device(args.gpu_id)

    seed_everything(args.seed)
    config = create_qwen3_config(
        attn_implementation="sdpa",
        num_experts=8,
        num_experts_per_tok=2,
        hidden_size=2048,
        num_attention_heads=32,
        num_key_value_heads=4,
    )

    attn_stage = Qwen3AttentionStage(config, dtype=torch_dtype).to(device)
    moe_stage = Qwen3MoEStageWithHead(config, args.num_classes, dtype=torch_dtype).to(device)
    attn_stage.eval()
    moe_stage.eval()

    # Generate deterministic synthetic inputs in fp32 then cast.
    seed_everything(args.seed + 1)
    attn_input = torch.randn(
        args.batch_size,
        args.seq_len,
        config.hidden_size,
        device=device,
        dtype=torch.float32,
    ).to(torch_dtype)
    moe_input = torch.randn(
        args.batch_size,
        args.seq_len,
        config.hidden_size,
        device=device,
        dtype=torch.float32,
    ).to(torch_dtype)
    labels = torch.randint(0, args.num_classes, (args.batch_size,), device=device)
    return attn_stage, moe_stage, attn_input, moe_input, labels


def _run_case(args: argparse.Namespace, case_id: str, baseline_case_id: str) -> tuple[dict[str, Any], int]:
    ensure_dirs(args.output_dir)
    device = torch.device(f"cuda:{args.gpu_id}")
    attn_stage, moe_stage, attn_input, moe_input, labels = _build_models_and_inputs(args)
    loss_fn = nn.CrossEntropyLoss()

    total_iters = args.warmup_iters + args.timed_iters
    timed_step_ms: list[float] = []
    timed_attn_ms: list[float] = []
    timed_moe_ms: list[float] = []
    timed_overlap_ms: list[float] = []

    first_nonfinite: dict[str, Any] | None = None
    all_finite = True

    last_attn_out: torch.Tensor | None = None
    last_moe_logits: torch.Tensor | None = None
    last_loss: torch.Tensor | None = None

    attn_stream = torch.cuda.Stream(device=device)
    moe_stream = torch.cuda.Stream(device=device)

    for iter_idx in range(total_iters):
        seed_everything(args.seed + iter_idx)
        attn_stage.zero_grad(set_to_none=True)
        moe_stage.zero_grad(set_to_none=True)

        step_start = time.perf_counter()
        attn_start = torch.cuda.Event(enable_timing=True)
        attn_end = torch.cuda.Event(enable_timing=True)
        moe_start = torch.cuda.Event(enable_timing=True)
        moe_end = torch.cuda.Event(enable_timing=True)
        fwd_window_start = torch.cuda.Event(enable_timing=True)
        fwd_window_end = torch.cuda.Event(enable_timing=True)

        if args.mode == "serial":
            fwd_window_start.record()
            attn_start.record()
            attn_out = attn_stage(attn_input)
            attn_end.record()

            moe_start.record()
            moe_logits = moe_stage(moe_input)
            moe_end.record()
            fwd_window_end.record()
        else:
            kickoff = torch.cuda.Event(enable_timing=False)
            fwd_window_start.record()
            with torch.cuda.stream(attn_stream):
                attn_stream.wait_event(kickoff)
                attn_start.record(attn_stream)
                attn_out = attn_stage(attn_input)
                attn_end.record(attn_stream)
            with torch.cuda.stream(moe_stream):
                moe_stream.wait_event(kickoff)
                moe_start.record(moe_stream)
                moe_logits = moe_stage(moe_input)
                moe_end.record(moe_stream)
            kickoff.record(torch.cuda.current_stream(device))
            torch.cuda.current_stream(device).wait_event(attn_end)
            torch.cuda.current_stream(device).wait_event(moe_end)
            fwd_window_end.record()

        loss_attn = attn_out.float().mean()
        loss_moe = loss_fn(moe_logits.float(), labels)
        loss = loss_attn + loss_moe
        loss.backward()

        torch.cuda.synchronize(device)
        step_end = time.perf_counter()

        attn_ms = float(attn_start.elapsed_time(attn_end))
        moe_ms = float(moe_start.elapsed_time(moe_end))
        fwd_combined_ms = float(fwd_window_start.elapsed_time(fwd_window_end))
        step_ms = (step_end - step_start) * 1000.0
        overlap_ms = max(0.0, attn_ms + moe_ms - fwd_combined_ms)

        is_timed = iter_idx >= args.warmup_iters
        if is_timed:
            timed_step_ms.append(step_ms)
            timed_attn_ms.append(attn_ms)
            timed_moe_ms.append(moe_ms)
            timed_overlap_ms.append(overlap_ms)

        if first_nonfinite is None:
            first_nonfinite = first_nonfinite_in_tensors(
                iter_idx=iter_idx,
                module="same_process",
                phase="forward",
                named_tensors=[("attn_output", attn_out), ("moe_logits", moe_logits), ("loss", loss)],
            )
        if first_nonfinite is None:
            first_nonfinite = first_nonfinite_in_tensors(
                iter_idx=iter_idx,
                module="same_process",
                phase="backward",
                named_tensors=_collect_grad_tensors(attn_stage, moe_stage),
            )
        if first_nonfinite is not None:
            all_finite = False

        last_attn_out = attn_out.detach()
        last_moe_logits = moe_logits.detach()
        last_loss = loss.detach()

    attn_sample = None
    if last_attn_out is not None:
        attn_sample = last_attn_out.flatten()[: args.attn_sample_elems].detach().cpu()
    moe_logits_cpu = last_moe_logits.detach().cpu() if last_moe_logits is not None else None

    finite = {
        "all_finite": bool(all_finite),
        "first_nonfinite": first_nonfinite,
    }

    timing_ms = {
        "step_total": float(sum(timed_step_ms) / len(timed_step_ms)) if timed_step_ms else None,
        "attn_cuda": float(sum(timed_attn_ms) / len(timed_attn_ms)) if timed_attn_ms else None,
        "moe_cuda": float(sum(timed_moe_ms) / len(timed_moe_ms)) if timed_moe_ms else None,
    }

    overlap_required = args.mode == "overlap"
    overlap_host_ms = float(sum(timed_overlap_ms) / len(timed_overlap_ms)) if timed_overlap_ms else 0.0
    overlap_valid = True
    status = "ok"
    exit_code = 0
    if overlap_required:
        overlap_valid = overlap_host_ms >= args.min_host_overlap_ms
        if not overlap_valid:
            status = "invalid_overlap"
            exit_code = 2

    overlap = {
        "required": overlap_required,
        "host_enqueue_overlap_ms": overlap_host_ms,
        "min_required_host_overlap_ms": float(args.min_host_overlap_ms if overlap_required else 0.0),
        "valid": bool(overlap_valid),
    }

    atol, rtol = tolerance_for_dtype(args.dtype)
    numeric_diff = {
        "baseline_case_id": baseline_case_id if args.mode == "overlap" else None,
        "baseline_metadata_match": args.mode == "serial",
        "max_abs_diff": None,
        "relative_error": None,
        "atol": float(atol),
        "rtol": float(rtol),
    }

    baseline_meta = _baseline_metadata(args)
    grad_sums = grad_checksums({"attn": attn_stage, "moe": moe_stage})

    if args.mode == "serial":
        artifact = {
            "metadata": baseline_meta,
            "outputs": {
                "attn_output_sample": attn_sample,
                "moe_logits": moe_logits_cpu,
            },
            "loss": float(last_loss.item()) if last_loss is not None else None,
            "grad_checksums": grad_sums,
        }
        save_baseline_artifact(
            output_dir=args.output_dir,
            baseline_case_id=baseline_case_id,
            artifact=artifact,
            baseline_override_path=args.baseline_json,
        )
    else:
        try:
            _baseline_path, artifact = load_baseline_artifact(
                output_dir=args.output_dir,
                baseline_case_id=baseline_case_id,
                baseline_override_path=args.baseline_json,
            )
            metadata_match = _metadata_match(baseline_meta, artifact.get("metadata", {}))
            numeric_diff["baseline_metadata_match"] = bool(metadata_match)
            numeric_diff["baseline_case_id"] = baseline_case_id
            if not metadata_match:
                status = "invalid_baseline"
                exit_code = 2
            else:
                max_abs_diff, rel_err = tensor_numeric_diff(
                    {
                        "attn_output_sample": attn_sample,
                        "moe_logits": moe_logits_cpu,
                    },
                    artifact.get("outputs", {}),
                )
                numeric_diff["max_abs_diff"] = max_abs_diff
                numeric_diff["relative_error"] = rel_err

                baseline_loss = artifact.get("loss")
                if baseline_loss is not None and last_loss is not None:
                    loss_abs_diff = abs(float(last_loss.item()) - float(baseline_loss))
                    loss_rel = loss_abs_diff / max(abs(float(baseline_loss)), 1e-12)
                    if numeric_diff["max_abs_diff"] is None:
                        numeric_diff["max_abs_diff"] = loss_abs_diff
                        numeric_diff["relative_error"] = loss_rel
                    else:
                        numeric_diff["max_abs_diff"] = max(float(numeric_diff["max_abs_diff"]), loss_abs_diff)
                        numeric_diff["relative_error"] = max(float(numeric_diff["relative_error"]), loss_rel)
        except Exception as exc:
            status = "invalid_baseline"
            exit_code = 2
            numeric_diff["baseline_metadata_match"] = False
            return (
                build_case_payload(
                    case_id=case_id,
                    experiment="same_process",
                    mode=args.mode,
                    dtype=args.dtype,
                    seq_len=args.seq_len,
                    seed=args.seed,
                    gpu_id=args.gpu_id,
                    use_mps=False,
                    status=status,
                    finite=finite,
                    numeric_diff=numeric_diff,
                    timing_ms=timing_ms,
                    overlap=overlap,
                    error={
                        "code": "invalid_baseline",
                        "message": str(exc),
                        "traceback": None,
                    },
                    metadata={
                        "grad_checksums": grad_sums,
                        "batch_size": args.batch_size,
                        "num_classes": args.num_classes,
                        "warmup_iters": args.warmup_iters,
                        "timed_iters": args.timed_iters,
                    },
                ),
                exit_code,
            )

    payload = build_case_payload(
        case_id=case_id,
        experiment="same_process",
        mode=args.mode,
        dtype=args.dtype,
        seq_len=args.seq_len,
        seed=args.seed,
        gpu_id=args.gpu_id,
        use_mps=False,
        status=status,
        finite=finite,
        numeric_diff=numeric_diff,
        timing_ms=timing_ms,
        overlap=overlap,
        metadata={
            "grad_checksums": grad_sums,
            "batch_size": args.batch_size,
            "num_classes": args.num_classes,
            "warmup_iters": args.warmup_iters,
            "timed_iters": args.timed_iters,
            "baseline_metadata": baseline_meta,
        },
    )
    return payload, exit_code


def main() -> int:
    args = _parse_args()
    case_id = build_case_id(
        experiment="same_process",
        mode=args.mode,
        use_mps=False,
        dtype=args.dtype,
        seq_len=args.seq_len,
        seed=args.seed,
        gpu_id=args.gpu_id,
    )
    baseline_case_id = build_case_id(
        experiment="same_process",
        mode="serial",
        use_mps=False,
        dtype=args.dtype,
        seq_len=args.seq_len,
        seed=args.seed,
        gpu_id=args.gpu_id,
    )

    ok, preflight_payload = preflight_checks(
        experiment="same_process",
        mode=args.mode,
        gpu_id=args.gpu_id,
        dtype=args.dtype,
        seq_len=args.seq_len,
        seed=args.seed,
        use_mps=False,
        output_dir=args.output_dir,
        strict_schema=args.strict_schema,
    )
    if not ok:
        ensure_dirs(args.output_dir)
        write_case_json(
            preflight_payload,
            output_dir=args.output_dir,
            json_output=args.json_output,
            strict_schema=args.strict_schema,
        )
        return 2

    try:
        payload, code = _run_case(args, case_id, baseline_case_id)
    except Exception as exc:
        payload = make_error_payload(
            case_id=case_id,
            experiment="same_process",
            mode=args.mode,
            dtype=args.dtype,
            seq_len=args.seq_len,
            seed=args.seed,
            gpu_id=args.gpu_id,
            use_mps=False,
            status="runtime_error",
            code="runtime_error",
            message=str(exc),
        )
        code = 1

    ensure_dirs(args.output_dir)
    write_case_json(
        payload,
        output_dir=args.output_dir,
        json_output=args.json_output,
        strict_schema=args.strict_schema,
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
