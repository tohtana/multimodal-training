import pytest
from pydantic import ValidationError

from python.pipeline.run_artifact_schema import (
    BlockerRow,
    EdgeRow,
    LayerTruncationRow,
    MetricsRow,
    RunRow,
    StageRow,
    TrainingRow,
    project_summary,
    SUMMARY_PROJECTION,
)


def _base_payload(status="ok"):
    blocker = None
    if status == "blocked":
        blocker = BlockerRow(
            category="asymmetric_routing_unsupported",
            reason="Backward redistribution is missing.",
            code_path="multimodal-training/python/pipeline/vlm_runner.py:153",
            code_path_git_sha="abc123",
            error_excerpt='"Asymmetric real-VLM backward routing is not implemented"',
            config_diff="vision.num_gpus: 4 -> 2",
            next_action="Implement gradient aggregation.",
        )
    return {
        "run_id": f"run-{status}",
        "status": status,
        "cell": "collocated" if status != "blocked" else "asymmetric",
        "dry_run": status == "skipped",
        "framework_only": False,
        "model_id": "Qwen/Qwen3-VL-8B-Instruct",
        "dense": True,
        "git_sha": "abc123",
        "submodule_shas": {"multimodal-training": "def456"},
        "command": ["python", "script.py"],
        "started_utc": "2026-05-01T00:00:00Z",
        "finished_utc": "2026-05-01T00:01:00Z",
        "error_excerpt": "failure" if status in {"failed", "oom"} else None,
        "layer_truncation": LayerTruncationRow(
            vision_source_field="vision_config.depth",
            language_source_field="text_config.num_hidden_layers",
            vision_override_path="engine_config.megatron_num_layers",
            language_override_path="engine_config.megatron_num_layers",
            vision_effective_layers=8,
            language_effective_layers=8,
            vision_original_layers=32,
            language_original_layers=32,
        ),
        "stages": [
            StageRow(
                name="vision",
                resource_set="vision_gpus",
                actor_count=4,
                physical_gpu_ids={0: "gpu0", 1: "gpu1", 2: "gpu2", 3: "gpu3"},
                requested_device_ids=[0, 1, 2, 3],
                placement_match=True,
                process_group_world_size=4,
                megatron={"tp": 4, "cp": 1, "pp": 1, "ep": 1},
            )
        ],
        "edges": [
            EdgeRow(
                from_stage="vision",
                to_stage="text",
                transfer_tier="t1",
                routing_map={"num_src": 4, "num_dst": 4},
            )
        ],
        "training": TrainingRow(
            iterations=1,
            warmup_iterations=0,
            batch_size=1,
            optimizer_update_verified=status == "ok",
            iteration_step_counter_advanced=status == "ok",
            expected_optimizer_step_delta=1,
            actual_optimizer_step_delta=1 if status == "ok" else None,
        ),
        "metrics": MetricsRow(
            loss_finite=status == "ok",
            iter_time_ms_p50=10.0 if status == "ok" else None,
            iter_time_ms_p90=12.0 if status == "ok" else None,
            cuda_max_allocated_bytes=100 if status == "ok" else None,
            cuda_max_reserved_bytes=200 if status == "ok" else None,
        ),
        "blocker": blocker,
    }


@pytest.mark.cpu_only
@pytest.mark.parametrize("status", ["ok", "blocked", "failed", "oom", "skipped"])
def test_run_row_accepts_required_status_fixtures(status):
    row = RunRow(**_base_payload(status))
    assert row.status == status


@pytest.mark.cpu_only
@pytest.mark.parametrize("field", ["run_id", "status", "cell", "training", "metrics", "layer_truncation"])
def test_run_row_rejects_missing_required_fields(field):
    payload = _base_payload("ok")
    payload.pop(field)
    with pytest.raises(ValidationError):
        RunRow(**payload)


@pytest.mark.cpu_only
def test_blocked_row_requires_blocker():
    payload = _base_payload("blocked")
    payload["blocker"] = None
    with pytest.raises(ValidationError):
        RunRow(**payload)


@pytest.mark.cpu_only
def test_summary_projection_columns_resolve_on_run_row():
    row = RunRow(**_base_payload("blocked"))
    summary = project_summary(row)
    assert set(summary) == {name for name, _ in SUMMARY_PROJECTION}
    assert summary["blocker_category"] == "asymmetric_routing_unsupported"
