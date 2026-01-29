# Test Guide

The test suite is split into two categories using `pytest` markers:

- `cpu_only`: unit-level coverage that runs on a single process without CUDA/NCCL.
- `gpu` + `integration`: multi-GPU or end-to-end checks that rely on CUDA, NCCL, and in some cases optional datasets or pretrained weights.

## Quick CPU Regression Pass

```bash
pytest -m cpu_only
```

This runs the light-weight ragged utilities and dataset modality tests:

- `tests/test_sequence_parallel_ragged.py`
- `tests/test_ragged_sp_smoke.py`
- `tests/test_dataset_modalities.py`

These tests exercise the split/pad/unpad utilities and dataset preprocessing logic without requiring GPUs.

## GPU / Distributed Checks

The following scenarios require at least 2 GPUs with NCCL enabled. Launch them with `torchrun` (or another launcher that sets up the process group) so each rank runs the test module.

**IMPORTANT**: Run each GPU test file individually, not all together. Running multiple distributed test files in a single pytest invocation can cause hangs due to process group cleanup issues.

### Sequence-parallel collectives smoke

```bash
torchrun --nproc_per_node=2 -m pytest tests/test_split_gather.py -m gpu -v
```

Verifies the ragged-safe all-gather across ranks.

### Vision model parity / diagnostics

```bash
# Tests vision sequence parallelism with pretrained weights
torchrun --nproc_per_node=2 -m pytest tests/test_vision_compare.py -m gpu -v

# Detailed diagnostic test for vision sequence parallelism
torchrun --nproc_per_node=2 -m pytest tests/test_vision_detailed.py -m gpu -v
```

These tests load the Qwen vision encoder (optionally with pretrained weights) and compare outputs across SP configurations. They expect GPUs with BF16 support and access to the `Qwen/Qwen2.5-VL-3B-Instruct` weights (cached locally via Hugging Face).

### Text model with DeepSpeed AutoTP

```bash
torchrun --nproc_per_node=2 -m pytest tests/deepspeed/test_text_autotp.py -v
```

Verifies that DeepSpeed AutoTP (tensor parallelism) produces identical losses and parameter updates compared to a non-parallel baseline. Uses a small Qwen text model (2 layers) and compares replicated parameters (layer norms) after each training step. Tests pure TP mode with ZeRO Stage 1.

### Text model with AutoTP + Data Parallel

```bash
torchrun --nproc_per_node=4 -m pytest tests/deepspeed/test_text_autotp_dp.py -v
```

Verifies hybrid parallelism combining DeepSpeed AutoTP with Data Parallel (tp_size=2, dp_size=2). Tests that different DP ranks process different data, TP ranks within each DP group synchronize correctly, and parameters are synchronized across DP ranks after optimizer steps.

### Vision model with DeepSpeed Sequence Parallel

```bash
torchrun --nproc_per_node=2 -m pytest tests/deepspeed/test_vision_sp.py -v
```

Verifies that DeepSpeed Sequence Parallel (SP) for the vision model produces identical losses and parameter updates compared to a non-parallel baseline. Uses a small Qwen vision model (2 layers) and compares replicated parameters (layer norms) after each training step. Tests pure SP mode (world_size == sp_size).

### Vision model with Sequence Parallel + Data Parallel

```bash
torchrun --nproc_per_node=4 -m pytest tests/deepspeed/test_vision_sp_dp.py -v
```

Verifies hybrid parallelism combining DeepSpeed Sequence Parallel with Data Parallel for the vision model (sp_size=2, dp_size=2). Tests that different DP ranks process different data, SP ranks within each DP group synchronize correctly, and parameters are synchronized across DP ranks after optimizer steps.

### Megatron single-trainer (text-only or vision-only)

```bash
MEGATRON_SINGLE_MODEL=Qwen/Qwen1.5-MoE-A2.7B-Chat \
MEGATRON_SINGLE_MODEL_TYPE=qwen2_moe \
MEGATRON_SINGLE_TRAINER=text \
MEGATRON_SINGLE_LOAD_WEIGHTS=0 \
pytest tests/megatron/test_single_trainer.py -m gpu
```

To run a TP/EP matrix with a single command:

```bash
MEGATRON_SINGLE_MODEL=Qwen/Qwen1.5-MoE-A2.7B-Chat \
MEGATRON_SINGLE_MODEL_TYPE=qwen2_moe \
MEGATRON_SINGLE_MATRIX=1 \
MEGATRON_SINGLE_LOAD_WEIGHTS=0 \
pytest tests/megatron/test_single_trainer.py -m gpu
```

Override the per-run topology with:

- `MEGATRON_SINGLE_TP_SIZE`
- `MEGATRON_SINGLE_EP_SIZE`
- `MEGATRON_SINGLE_NUM_ACTORS` (defaults to TP * EP)

### Megatron pre-PP readiness (Qwen3-VL MoE, 4-layer split)

```bash
HF_HOME=/mnt/local_storage/hf-cache \
pytest tests/megatron/test_engine_megatron_prepp.py -k test_megatron_engine_prepp -m gpu -v
```

Defaults baked into the test:

- `MEGATRON_TEST_MODEL=Qwen/Qwen3-VL-30B-A3B-Instruct`
- `MEGATRON_TEST_BRIDGE_LOAD_PATH=/mnt/local_storage/checkpoints/qwen3_vl_30b_a3b_4l_split`
- `MEGATRON_TEST_NUM_LAYERS=4`

If the split checkpoint is missing, the test generates it automatically via `scripts/split_checkpoint.py`
with `--save-hf-safetensors`.

Parallel sweep behavior:

- Vision: TP=4 (small) or TP=8 (full scale)
- Text: TP=4/8 (tp_tp) and EP=4/8 (tp_ep)

Requires 4 GPUs by default (8 GPUs if `MEGATRON_TEST_FULL_SCALE=1`).

Override sweep sizes and actor counts with:

- `MEGATRON_TEST_TP_SIZES=2,4` (tp_tp vision/text TP sizes)
- `MEGATRON_TEST_EP_SIZES=2,4` (tp_ep text EP sizes; vision TP uses `MEGATRON_TEST_TP_SIZES`)
- `MEGATRON_TEST_VISION_ACTORS` and `MEGATRON_TEST_TEXT_ACTORS` (must match world size)

## Dataset alignment test

`tests/test_dataset_modalities.py::test_real_dataset_alignment` automatically skips if the COCO validation set defined in `DEFAULT_DATA_REGISTRY` is not present. To exercise it fully, download the dataset referenced in the registry before running `pytest -m cpu_only`.

## Tips

- **CPU tests**: Run with `pytest -m cpu_only` (26 tests, all passing)
- **GPU tests**: Run each file individually with torchrun (do NOT run all together)
- Set `HF_HOME` if you want weights/datasets cached at a specific path
- For verbose logging during GPU runs, add `-s` and export `CUDA_LAUNCH_BLOCKING=1`
- Use `-v` flag for more detailed test output

## Summary

The latest test summary is generated at `tests/summary/SUMMARY.md`.
