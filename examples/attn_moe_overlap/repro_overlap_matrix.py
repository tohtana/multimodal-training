"""Matrix runner for same-process and two-process overlap repro cases."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from examples.attn_moe_overlap.repro_overlap_common import build_case_id


def _parse_list(text: str) -> list[str]:
    return [piece.strip() for piece in text.split(",") if piece.strip()]


def _parse_int_list(text: str) -> list[int]:
    return [int(piece.strip()) for piece in text.split(",") if piece.strip()]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run overlap minimal repro matrix")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--warmup-iters", type=int, default=2)
    parser.add_argument("--timed-iters", type=int, default=2)
    parser.add_argument("--seq-lens", type=str, default="512,1024,8192")
    parser.add_argument("--dtypes", type=str, default="bf16,fp32")
    parser.add_argument("--min-host-overlap-ms", type=float, default=0.25)
    parser.add_argument("--worker-timeout-s", type=float, default=120.0)
    parser.add_argument("--attn-thread-pct", type=int, default=None)
    parser.add_argument("--moe-thread-pct", type=int, default=None)
    parser.add_argument("--capture-nsys", choices=["on", "off"], default="off")
    parser.add_argument("--nsys-bin", type=str, default="nsys")
    parser.add_argument("--strict-schema", dest="strict_schema", action="store_true", default=True)
    parser.add_argument("--no-strict-schema", dest="strict_schema", action="store_false")
    return parser.parse_args()


def _build_command(case: dict[str, Any], args: argparse.Namespace) -> list[str]:
    base = [sys.executable, "-m"]
    if case["experiment"] == "same_process":
        module = "examples.attn_moe_overlap.repro_overlap_same_process"
        cmd = [
            *base,
            module,
            "--mode",
            case["mode"],
            "--gpu-id",
            str(args.gpu_id),
            "--seq-len",
            str(case["seq_len"]),
            "--dtype",
            case["dtype"],
            "--batch-size",
            str(args.batch_size),
            "--num-classes",
            str(args.num_classes),
            "--warmup-iters",
            str(args.warmup_iters),
            "--timed-iters",
            str(args.timed_iters),
            "--seed",
            str(args.seed),
            "--min-host-overlap-ms",
            str(args.min_host_overlap_ms),
            "--output-dir",
            args.output_dir,
        ]
    else:
        module = "examples.attn_moe_overlap.repro_overlap_two_process_mps"
        cmd = [
            *base,
            module,
            "--mode",
            case["mode"],
            "--use-mps",
            "on" if case["use_mps"] else "off",
            "--gpu-id",
            str(args.gpu_id),
            "--seq-len",
            str(case["seq_len"]),
            "--dtype",
            case["dtype"],
            "--batch-size",
            str(args.batch_size),
            "--num-classes",
            str(args.num_classes),
            "--warmup-iters",
            str(args.warmup_iters),
            "--timed-iters",
            str(args.timed_iters),
            "--seed",
            str(args.seed),
            "--min-host-overlap-ms",
            str(args.min_host_overlap_ms),
            "--worker-timeout-s",
            str(args.worker_timeout_s),
            "--output-dir",
            args.output_dir,
        ]
        if args.attn_thread_pct is not None:
            cmd.extend(["--attn-thread-pct", str(args.attn_thread_pct)])
        if args.moe_thread_pct is not None:
            cmd.extend(["--moe-thread-pct", str(args.moe_thread_pct)])

    if args.strict_schema:
        cmd.append("--strict-schema")
    else:
        cmd.append("--no-strict-schema")
    return cmd


def _load_case_payload(output_dir: str, case_id: str) -> dict[str, Any] | None:
    path = Path(output_dir) / "cases" / f"{case_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _case_row(case: dict[str, Any], cmd: list[str], returncode: int, payload: dict[str, Any] | None) -> dict[str, Any]:
    row = {
        "case_id": case["case_id"],
        "experiment": case["experiment"],
        "mode": case["mode"],
        "use_mps": case["use_mps"],
        "dtype": case["dtype"],
        "seq_len": case["seq_len"],
        "command": cmd,
        "returncode": returncode,
        "payload_exists": payload is not None,
    }
    if payload is not None:
        row.update(
            {
                "status": payload.get("status"),
                "all_finite": payload.get("finite", {}).get("all_finite"),
                "overlap_valid": payload.get("overlap", {}).get("valid"),
                "max_abs_diff": payload.get("numeric_diff", {}).get("max_abs_diff"),
                "relative_error": payload.get("numeric_diff", {}).get("relative_error"),
            }
        )
    else:
        row.update(
            {
                "status": "missing_payload",
                "all_finite": None,
                "overlap_valid": None,
                "max_abs_diff": None,
                "relative_error": None,
            }
        )
    return row


def _build_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    dtypes = _parse_list(args.dtypes)
    seq_lens = _parse_int_list(args.seq_lens)
    cases: list[dict[str, Any]] = []

    for dtype in dtypes:
        for seq_len in seq_lens:
            # Same-process baseline then overlap consumer
            for mode in ("serial", "overlap"):
                case = {
                    "experiment": "same_process",
                    "mode": mode,
                    "use_mps": False,
                    "dtype": dtype,
                    "seq_len": seq_len,
                }
                case["case_id"] = build_case_id(
                    experiment=case["experiment"],
                    mode=case["mode"],
                    use_mps=case["use_mps"],
                    dtype=case["dtype"],
                    seq_len=case["seq_len"],
                    seed=args.seed,
                    gpu_id=args.gpu_id,
                )
                cases.append(case)

            # Two-process MPS off baseline+parallel
            for use_mps in (False, True):
                for mode in ("serial", "parallel"):
                    case = {
                        "experiment": "two_process",
                        "mode": mode,
                        "use_mps": use_mps,
                        "dtype": dtype,
                        "seq_len": seq_len,
                    }
                    case["case_id"] = build_case_id(
                        experiment=case["experiment"],
                        mode=case["mode"],
                        use_mps=case["use_mps"],
                        dtype=case["dtype"],
                        seq_len=case["seq_len"],
                        seed=args.seed,
                        gpu_id=args.gpu_id,
                        attn_thread_pct=args.attn_thread_pct,
                        moe_thread_pct=args.moe_thread_pct,
                    )
                    cases.append(case)

    return cases


def _write_summary(output_dir: str, rows: list[dict[str, Any]], overall_status: str, nsys_status: str | None) -> None:
    summary = {
        "schema_version": "mps_overlap_repro.matrix.v1",
        "overall_status": overall_status,
        "nsys_status": nsys_status,
        "rows": rows,
    }
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "matrix_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    lines = []
    lines.append("# MPS Overlap Repro Matrix")
    lines.append("")
    lines.append(f"overall_status: `{overall_status}`")
    if nsys_status is not None:
        lines.append(f"nsys_status: `{nsys_status}`")
    lines.append("")
    lines.append(
        "| case_id | experiment | mode | use_mps | dtype | seq_len | status | all_finite | overlap_valid | max_abs_diff | relative_error | returncode |"
    )
    lines.append(
        "|---|---|---|---|---|---|---|---|---|---|---|---|"
    )
    for row in rows:
        lines.append(
            "| {case_id} | {experiment} | {mode} | {use_mps} | {dtype} | {seq_len} | {status} | {all_finite} | {overlap_valid} | {max_abs_diff} | {relative_error} | {returncode} |".format(
                case_id=row.get("case_id"),
                experiment=row.get("experiment"),
                mode=row.get("mode"),
                use_mps=row.get("use_mps"),
                dtype=row.get("dtype"),
                seq_len=row.get("seq_len"),
                status=row.get("status"),
                all_finite=row.get("all_finite"),
                overlap_valid=row.get("overlap_valid"),
                max_abs_diff=row.get("max_abs_diff"),
                relative_error=row.get("relative_error"),
                returncode=row.get("returncode"),
            )
        )

    (out_dir / "matrix_summary.md").write_text("\n".join(lines) + "\n")


def _select_nsys_pair(rows: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    failing: dict[str, Any] | None = None
    passing: dict[str, Any] | None = None

    for row in rows:
        status = row.get("status")
        all_finite = row.get("all_finite")
        overlap_valid = row.get("overlap_valid")
        if status != "ok" or (all_finite is False and bool(overlap_valid)):
            failing = row
            break

    if failing is None:
        return None, None

    for row in rows:
        if row.get("status") != "ok":
            continue
        if row.get("all_finite") is not True:
            continue
        if (
            row.get("experiment") == failing.get("experiment")
            and row.get("dtype") == failing.get("dtype")
            and row.get("seq_len") == failing.get("seq_len")
            and row.get("use_mps") == failing.get("use_mps")
        ):
            passing = row
            break

    if passing is None:
        for row in rows:
            if row.get("status") == "ok" and row.get("all_finite") is True:
                passing = row
                break

    return failing, passing


def _capture_nsys_pair(
    *,
    output_dir: str,
    nsys_bin: str,
    failing: dict[str, Any],
    passing: dict[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    if shutil.which(nsys_bin) is None and not Path(nsys_bin).exists():
        return "nsys_capture_failed", []

    nsys_dir = Path(output_dir) / "nsys"
    nsys_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, Any]] = []
    for row in (failing, passing):
        case_id = row["case_id"]
        out_prefix = nsys_dir / case_id
        cmd = [
            nsys_bin,
            "profile",
            "--force-overwrite=true",
            "-o",
            str(out_prefix),
            *row["command"],
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        entries.append(
            {
                "case_id": case_id,
                "command": cmd,
                "returncode": proc.returncode,
                "stdout_tail": proc.stdout[-4000:],
                "stderr_tail": proc.stderr[-4000:],
                "artifact_prefix": str(out_prefix),
            }
        )
        if proc.returncode != 0:
            trace_index = {
                "status": "nsys_capture_failed",
                "selected": {
                    "failing_case_id": failing["case_id"],
                    "passing_case_id": passing["case_id"],
                },
                "entries": entries,
            }
            (nsys_dir / "trace_index.json").write_text(json.dumps(trace_index, indent=2, sort_keys=True) + "\n")
            return "nsys_capture_failed", entries

    trace_index = {
        "status": "ok",
        "selected": {
            "failing_case_id": failing["case_id"],
            "passing_case_id": passing["case_id"],
        },
        "entries": entries,
    }
    (nsys_dir / "trace_index.json").write_text(json.dumps(trace_index, indent=2, sort_keys=True) + "\n")
    return "ok", entries


def main() -> int:
    args = _parse_args()
    out_dir = Path(args.output_dir)
    (out_dir / "cases").mkdir(parents=True, exist_ok=True)
    (out_dir / "baselines").mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    overall_status = "ok"

    for case in _build_cases(args):
        cmd = _build_command(case, args)
        proc = subprocess.run(cmd, capture_output=True, text=True)
        payload = _load_case_payload(args.output_dir, case["case_id"])
        row = _case_row(case, cmd, proc.returncode, payload)
        row["stdout_tail"] = proc.stdout[-4000:]
        row["stderr_tail"] = proc.stderr[-4000:]
        rows.append(row)

        if row["status"] != "ok" or row["returncode"] != 0:
            overall_status = "failed"

    nsys_status: str | None = None
    if args.capture_nsys == "on":
        failing, passing = _select_nsys_pair(rows)
        if failing is None or passing is None:
            nsys_status = "nsys_capture_failed"
            overall_status = "failed"
        else:
            nsys_status, _entries = _capture_nsys_pair(
                output_dir=args.output_dir,
                nsys_bin=args.nsys_bin,
                failing=failing,
                passing=passing,
            )
            if nsys_status != "ok":
                overall_status = "failed"

    _write_summary(args.output_dir, rows, overall_status, nsys_status)
    return 0 if overall_status == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
