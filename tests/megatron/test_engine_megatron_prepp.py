"""Pre-PP readiness test for Megatron engine initialization."""

import json
import math
import subprocess
import sys
import time
from pathlib import Path

import pytest
import ray
import torch

pytestmark = [pytest.mark.gpu]

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MEGATRON_ROOT = PROJECT_ROOT / "Megatron-LM"
MS_SWIFT_ROOT = PROJECT_ROOT / "ms-swift"

sys.path.insert(0, str(PROJECT_ROOT / "multimodal-training"))
sys.path.insert(0, str(MEGATRON_ROOT))
sys.path.insert(0, str(MS_SWIFT_ROOT))

from python.ray.actor_group import ActorGroup  # noqa: E402
from python.ray.megatron_trainer import (  # noqa: E402
    MegatronTextTrainer,
    MegatronVisionTrainer,
)


def _parse_sizes(raw_value: str | None, default_sizes: list[int]) -> list[int]:
    if not raw_value:
        return default_sizes
    sizes: list[int] = []
    for value in raw_value.split(","):
        value = value.strip()
        if value:
            sizes.append(int(value))
    return sizes or default_sizes


def _resolve_actor_count(env_key: str, expected: int) -> int:
    import os

    raw_value = os.environ.get(env_key)
    if raw_value is None:
        return expected
    count = int(raw_value)
    if count != expected:
        raise ValueError(f"{env_key} must match the expected world size ({expected}).")
    return count


def _build_component_config(model_path: str, component: str, engine_overrides: dict | None = None):
    import os

    expert_model_parallel_size = int(os.environ.get("MEGATRON_TEST_EP_SIZE", "1"))
    num_experts_env = os.environ.get("MEGATRON_TEST_NUM_EXPERTS")
    num_layers_env = os.environ.get("MEGATRON_TEST_NUM_LAYERS")
    load_weights_env = os.environ.get("MEGATRON_TEST_LOAD_WEIGHTS", "true").lower()
    if component:
        scoped = os.environ.get(f"MEGATRON_TEST_LOAD_WEIGHTS_{component.upper()}")
        if scoped is not None:
            load_weights_env = scoped.lower()
    load_weights = load_weights_env not in {"0", "false", "no"}
    bridge_load_path = os.environ.get("MEGATRON_TEST_BRIDGE_LOAD_PATH")
    model_type = os.environ.get("MEGATRON_TEST_MODEL_TYPE")
    if not model_type:
        if any(tag in model_path for tag in ("A3B", "A22B")):
            model_type = "qwen3_moe_vl"
        else:
            model_type = "qwen3_vl"
    config = {
        "model_name": model_path,
        "model_type": model_type,
        "engine": "megatron",
        "engine_config": {
            "tensor_parallel_size": 1,
            "sequence_parallel_size": 1,
            "pipeline_model_parallel_size": 1,
            "attention_backend": "unfused",
            "expert_model_parallel_size": expert_model_parallel_size,
            "load_weights": load_weights,
        },
        "parallelism": "tensor",
        "dtype": "bfloat16",
        "attention_backend": "sdpa",
        "activation_checkpointing": False,
        "autocast": False,
        "seed": 123,
        "dp_size": 1,
        "parallel_size": 1,
        "text_seq_len": 4,
    }
    if bridge_load_path:
        config["engine_config"]["bridge_load_path"] = bridge_load_path
    if num_experts_env is not None:
        config["engine_config"]["num_experts"] = int(num_experts_env)
    if num_layers_env is not None:
        config["engine_config"]["megatron_num_layers"] = int(num_layers_env)
    if engine_overrides:
        config["engine_config"].update(engine_overrides)
    return config


def _default_bridge_path(full_scale: bool) -> str:
    if full_scale:
        shm_path = Path("/dev/shm/qwen3_vl_30b_a3b_split")
        local_path = Path("/mnt/local_storage/checkpoints/qwen3_vl_30b_a3b_split")
    else:
        shm_path = Path("/dev/shm/qwen3_vl_30b_a3b_4l_split")
        local_path = Path("/mnt/local_storage/checkpoints/qwen3_vl_30b_a3b_4l_split")
    return str(shm_path if shm_path.exists() else local_path)


def _load_model_config(bridge_load_path: str | None) -> dict | None:
    if not bridge_load_path:
        return None
    config_path = Path(bridge_load_path) / "config.json"
    if not config_path.exists():
        return None
    with open(config_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _infer_num_query_groups(model_config: dict | None) -> int | None:
    if not model_config:
        return None
    text_config = model_config.get("text_config") or {}
    return text_config.get("num_query_groups") or text_config.get("num_key_value_heads")


def _log(message: str, log_path: str | None = None) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[prepp {timestamp}] {message}"
    print(line, flush=True)
    if log_path:
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(f"{line}\n")
            handle.flush()


def _execute_all_with_timeout(
    group: ActorGroup, label: str, timeout_s: int, *args, log_path: str | None = None, **kwargs
):
    _log(f"{label} start (timeout={timeout_s}s)", log_path=log_path)
    refs = group.execute_all_async(label, *args, **kwargs)
    results = ray.get(refs, timeout=timeout_s)
    _log(f"{label} done", log_path=log_path)
    return results


def _setup_faulthandler(stack_log_path: str | None) -> None:
    if not stack_log_path:
        return
    import faulthandler
    import signal

    Path(stack_log_path).parent.mkdir(parents=True, exist_ok=True)
    stack_file = open(stack_log_path, "a", encoding="utf-8")
    faulthandler.enable(all_threads=True)
    faulthandler.register(signal.SIGUSR1, file=stack_file, all_threads=True)
    _log(f"Faulthandler enabled; send SIGUSR1 to dump stacks -> {stack_log_path}", log_path=stack_log_path)


@pytest.mark.parametrize("verify_weights", [False, True])
@pytest.mark.parametrize("parallel_case", ["tp_tp", "tp_ep"])
def test_megatron_engine_prepp(verify_weights: bool, parallel_case: str):
    import os

    full_scale = os.environ.get("MEGATRON_TEST_FULL_SCALE", "0").lower() in {"1", "true", "yes"}
    focus_case = os.environ.get("MEGATRON_TEST_ONLY_CASE")
    if focus_case and focus_case != parallel_case:
        pytest.skip(f"Skipping parallel_case={parallel_case}; focus on {focus_case}.")
    focus_verify = os.environ.get("MEGATRON_TEST_ONLY_VERIFY")
    if focus_verify is not None:
        focus_verify_bool = focus_verify.lower() in {"1", "true", "yes"}
        if verify_weights != focus_verify_bool:
            pytest.skip(f"Skipping verify_weights={verify_weights}; focus on {focus_verify_bool}.")

    os.environ.setdefault("MEGATRON_TEST_MODEL", "Qwen/Qwen3-VL-30B-A3B-Instruct")
    os.environ.setdefault(
        "MEGATRON_TEST_BRIDGE_LOAD_PATH",
        _default_bridge_path(full_scale),
    )
    if not full_scale:
        os.environ.setdefault("MEGATRON_TEST_NUM_LAYERS", "4")

    arch_list = None
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        arch_list = f"{major}.{minor}"
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", arch_list)
    try:
        import megatron  # noqa: F401
    except Exception:
        pytest.skip("Megatron-LM is not available; skipping Megatron pre-PP readiness test.")

    model_path = os.environ.get("MEGATRON_TEST_MODEL")
    if not model_path:
        pytest.skip("Set MEGATRON_TEST_MODEL to a HF model path for Megatron pre-PP readiness test.")

    bridge_load_path = os.environ.get("MEGATRON_TEST_BRIDGE_LOAD_PATH")
    log_path = os.environ.get("MEGATRON_TEST_DEBUG_LOG")
    stack_log_path = os.environ.get("MEGATRON_TEST_STACK_LOG")
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    _setup_faulthandler(stack_log_path)

    _log(
        "Debug start: "
        f"case={parallel_case} verify_weights={verify_weights} full_scale={full_scale} "
        f"bridge_load_path={bridge_load_path}",
        log_path=log_path,
    )

    if bridge_load_path and not Path(bridge_load_path).exists() and not full_scale:
        _log(f"Generating 4-layer split checkpoint at {bridge_load_path}", log_path=log_path)
        split_script = PROJECT_ROOT / "multimodal-training" / "scripts" / "split_checkpoint.py"
        split_cmd = [
            sys.executable,
            str(split_script),
            "--model-name",
            model_path,
            "--model-type",
            "qwen3_vl",
            "--num-text-layers",
            "4",
            "--save-hf-safetensors",
            "--output-dir",
            bridge_load_path,
        ]
        env = os.environ.copy()
        env.setdefault("HF_HOME", "/mnt/local_storage/hf-cache")
        subprocess.check_call(split_cmd, env=env, cwd=str(PROJECT_ROOT / "multimodal-training"))

    default_size = 8 if full_scale else 4
    step_timeout_s = int(os.environ.get("MEGATRON_TEST_STEP_TIMEOUT", "600"))
    if parallel_case == "tp_tp":
        sizes = _parse_sizes(os.environ.get("MEGATRON_TEST_TP_SIZES"), [default_size])
        pairs: list[tuple[int, int]] = [(size, size) for size in sizes]
    elif parallel_case == "tp_ep":
        ep_sizes = _parse_sizes(os.environ.get("MEGATRON_TEST_EP_SIZES"), [default_size])
        vision_tp_sizes = _parse_sizes(os.environ.get("MEGATRON_TEST_TP_SIZES"), [default_size])
        if len(vision_tp_sizes) == 1:
            vision_tp_sizes = vision_tp_sizes * len(ep_sizes)
        if len(vision_tp_sizes) != len(ep_sizes):
            raise ValueError("MEGATRON_TEST_TP_SIZES must be length 1 or match EP sizes for tp_ep.")
        pairs = list(zip(vision_tp_sizes, ep_sizes))
    else:
        raise ValueError(f"Unknown parallel_case: {parallel_case}")

    for vision_tp, text_ep in pairs:
        _log(
            f"Starting case={parallel_case} vision_tp={vision_tp} text_ep={text_ep} full_scale={full_scale}",
            log_path=log_path,
        )
        ray.init(
            address="auto",
            ignore_reinit_error=True,
            include_dashboard=False,
            runtime_env={
                "working_dir": str(PROJECT_ROOT / "multimodal-training"),
                "py_modules": [str(MEGATRON_ROOT), str(MS_SWIFT_ROOT)],
                "excludes": [".git/**", "**/.git/**", "**/__pycache__/**"],
                "env_vars": {
                    "PYTHONPATH": ":".join(
                        [
                            str(PROJECT_ROOT / "multimodal-training"),
                            str(MEGATRON_ROOT),
                            str(MS_SWIFT_ROOT),
                        ]
                    ),
                    "USE_HF": "1",
                    "HF_HOME": os.environ.get("HF_HOME", "/mnt/local_storage/hf-cache"),
                    **({"TORCH_CUDA_ARCH_LIST": arch_list} if arch_list else {}),
                },
            },
        )
        try:
            vision_overrides = {"tensor_parallel_size": vision_tp}
            if parallel_case == "tp_tp":
                text_overrides = {"tensor_parallel_size": vision_tp, "expert_model_parallel_size": 1}
            elif parallel_case == "tp_ep":
                text_overrides = {"tensor_parallel_size": 1, "expert_model_parallel_size": text_ep}
            else:
                raise ValueError(f"Unknown parallel_case: {parallel_case}")

            model_config = _load_model_config(bridge_load_path)
            num_query_groups = _infer_num_query_groups(model_config)
            if num_query_groups is not None:
                text_tp = text_overrides["tensor_parallel_size"]
                vision_tp = vision_overrides["tensor_parallel_size"]
                if num_query_groups % text_tp != 0 or num_query_groups % vision_tp != 0:
                    pytest.skip(
                        "num_query_groups "
                        f"({num_query_groups}) must be a multiple of text TP ({text_tp}) "
                        f"and vision TP ({vision_tp}); set MEGATRON_TEST_TP_SIZES to a divisor "
                        "(e.g. 4 for Qwen3-VL A3B)."
                    )

            collocate = os.environ.get("MEGATRON_TEST_COLLOCATE", "0").lower() in {"1", "true", "yes"}
            vision_config = _build_component_config(model_path, "vision", engine_overrides=vision_overrides)
            text_config = _build_component_config(model_path, "text", engine_overrides=text_overrides)

            vision_name = getattr(MegatronVisionTrainer, "__name__", MegatronVisionTrainer.__class__.__name__)
            text_name = getattr(MegatronTextTrainer, "__name__", MegatronTextTrainer.__class__.__name__)
            _log(f"Trainer classes: vision={vision_name} text={text_name}", log_path=log_path)

            expected_vision_actors = vision_overrides["tensor_parallel_size"]
            expected_text_actors = (
                text_overrides["tensor_parallel_size"] * text_overrides["expert_model_parallel_size"]
            )
            vision_actors = _resolve_actor_count("MEGATRON_TEST_VISION_ACTORS", expected_vision_actors)
            text_actors = _resolve_actor_count("MEGATRON_TEST_TEXT_ACTORS", expected_text_actors)

            if collocate and vision_actors != text_actors:
                pytest.skip(
                    "Collocation requires matching vision/text actor counts. "
                    "For tp_ep, set vision TP to match text EP, or disable collocation and "
                    "run on >= (vision TP + text EP) GPUs."
                )

            vision_group = ActorGroup(
                vision_config,
                MegatronVisionTrainer,
                num_actors=vision_actors,
                num_cpus=2,
                num_gpus=1,
                collocate=collocate,
            )
            text_group = ActorGroup(
                text_config,
                MegatronTextTrainer,
                num_actors=text_actors,
                num_cpus=2,
                num_gpus=1,
                collocate=collocate,
                placement_group_handle=vision_group.placement_group if collocate else None,
            )

            _execute_all_with_timeout(vision_group, "build_model", step_timeout_s, log_path=log_path)
            _execute_all_with_timeout(text_group, "build_model", step_timeout_s, log_path=log_path)
            _execute_all_with_timeout(vision_group, "initialize_trainer", step_timeout_s, log_path=log_path)
            _execute_all_with_timeout(text_group, "initialize_trainer", step_timeout_s, log_path=log_path)

            vision_pg = _execute_all_with_timeout(
                vision_group, "is_process_group_initialized", step_timeout_s, log_path=log_path
            )
            text_pg = _execute_all_with_timeout(
                text_group, "is_process_group_initialized", step_timeout_s, log_path=log_path
            )
            assert all(vision_pg), "Vision process group was not initialized"
            assert all(text_pg), "Text process group was not initialized"

            vision_outputs = _execute_all_with_timeout(
                vision_group, "forward_step", step_timeout_s, 0, log_path=log_path
            )
            if len(vision_outputs) != text_actors:
                raise RuntimeError(
                    "Vision outputs must align with text actors "
                    f"({len(vision_outputs)} != {text_actors})."
                )
            text_forward = _execute_all_with_timeout(
                text_group,
                "forward_step",
                step_timeout_s,
                vision_outputs,
                [0] * text_actors,
                log_path=log_path,
            )

            if verify_weights:
                status_list = _execute_all_with_timeout(
                    text_group, "get_weight_load_status", step_timeout_s, log_path=log_path
                )
                for status in status_list:
                    if not status["path_exists"]:
                        pytest.skip("MEGATRON_TEST_BRIDGE_LOAD_PATH not found; skipping weight-load verification.")
                    assert status["requested"], "Weight loading was not requested."
                    assert status["loaded"], "Weight loading did not complete successfully."
                for output in text_forward:
                    loss_value = output.get("loss")
                    assert loss_value is not None, "Missing loss from text forward output."
                    assert math.isfinite(loss_value), f"Non-finite loss after weight loading: {loss_value}"

            text_backward = _execute_all_with_timeout(text_group, "backward_step", step_timeout_s, log_path=log_path)
            _execute_all_with_timeout(
                vision_group, "backward_step", step_timeout_s, text_backward, log_path=log_path
            )
            _log(
                f"Completed case={parallel_case} vision_tp={vision_tp} text_ep={text_ep}",
                log_path=log_path,
            )
        finally:
            ray.shutdown()


def test_megatron_num_layers_override():
    import os

    try:
        import megatron  # noqa: F401
    except Exception:
        pytest.skip("Megatron-LM is not available; skipping Megatron num_layers override test.")

    model_path = os.environ.get("MEGATRON_TEST_MODEL")
    if not model_path:
        pytest.skip("Set MEGATRON_TEST_MODEL to a HF model path for Megatron num_layers override test.")

    ray.init(
        address="auto",
        ignore_reinit_error=True,
        include_dashboard=False,
        runtime_env={
            "working_dir": str(PROJECT_ROOT / "multimodal-training"),
            "py_modules": [str(MEGATRON_ROOT), str(MS_SWIFT_ROOT)],
            "excludes": [".git/**", "**/.git/**", "**/__pycache__/**"],
            "env_vars": {
                "PYTHONPATH": ":".join(
                    [
                        str(PROJECT_ROOT / "multimodal-training"),
                        str(MEGATRON_ROOT),
                        str(MS_SWIFT_ROOT),
                    ]
                ),
                "USE_HF": "1",
                "HF_HOME": os.environ.get("HF_HOME", "/mnt/local_storage/hf-cache"),
            },
        },
    )
    try:
        text_config = _build_component_config(
            model_path,
            engine_overrides={
                "load_weights": False,
                "megatron_num_layers": 2,
            },
        )
        text_group = ActorGroup(text_config, MegatronTextTrainer, num_actors=1, num_cpus=2, num_gpus=1)
        text_group.execute_all("build_model")
        num_layers = text_group.execute_all("get_megatron_num_layers")
        assert num_layers == [2], f"Expected num_layers override to be 2, got {num_layers}"
    finally:
        ray.shutdown()
