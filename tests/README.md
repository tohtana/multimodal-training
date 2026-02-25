# Test Guide

The test suite is split into two categories using `pytest` markers:

- `cpu_only`: unit-level coverage that runs on a single process without CUDA/NCCL.
- `gpu`: multi-GPU or end-to-end checks that rely on CUDA, NCCL, and in some cases optional datasets or pretrained weights.

## Quick CPU Regression Pass

```bash
pytest -m cpu_only
```

This runs all CPU-only tests including:
- Pipeline DAG validation, scheduling, config parsing, payload contracts, layout adapters
- Dataset modality tests
- Checkpoint script tests

## GPU / Distributed Checks

**IMPORTANT**: Run each GPU test file individually, not all together. Running multiple distributed test files in a single pytest invocation can cause hangs due to process group cleanup issues.

### Pipeline Tests

The pipeline framework tests cover all milestones (M0–M12):

```bash
# Pipeline DAG, scheduling, config parsing (CPU)
pytest tests/test_pipeline_dag.py -m cpu_only -v

# Payload contract tests (CPU)
pytest tests/test_pipeline_payload_contract.py -m cpu_only -v

# Layout adapter tests (CPU)
pytest tests/test_pipeline_layout.py -m cpu_only -v

# MLP pipeline on single GPU (M1)
pytest tests/test_pipeline_mlp_local.py -m gpu -v

# MLP pipeline with Ray actors (M2)
pytest tests/test_pipeline_mlp_ray.py -m gpu -v

# Cross-GPU transport (M3)
pytest tests/test_pipeline_cross_gpu.py -m gpu -v

# DeepSpeed engine integration (M4)
pytest tests/test_pipeline_deepspeed.py -m gpu -v

# VLM integration + legacy parity (M6)
pytest tests/test_pipeline_vlm.py -m gpu -v

# Asymmetric GPU allocation (M7)
pytest tests/test_pipeline_asymmetric.py -m gpu -v

# Three-stage pipeline with bridge (M9)
pytest tests/test_pipeline_three_stage.py -m gpu -v

# Bridge trainer tests
pytest tests/test_pipeline_bridge.py -m gpu -v

# 1F1B scheduling (M10)
pytest tests/test_pipeline_1f1b.py -m gpu -v

# Overlapping GPU sets (M11)
pytest tests/test_pipeline_overlap.py -m gpu -v

# Interleaved MoE (M12)
pytest tests/test_pipeline_moe_interleaved.py -m gpu -v
```

### DeepSpeed Engine Tests

```bash
# DeepSpeed pre-PP readiness
pytest tests/deepspeed/test_engine_deepspeed_prepp.py -m gpu -v
```

### Megatron Engine Tests

```bash
# Megatron single-trainer (text-only or vision-only)
MEGATRON_SINGLE_MODEL=Qwen/Qwen1.5-MoE-A2.7B-Chat \
MEGATRON_SINGLE_MODEL_TYPE=qwen2_moe \
MEGATRON_SINGLE_MATRIX=1 \
MEGATRON_SINGLE_LOAD_WEIGHTS=0 \
pytest tests/megatron/test_single_trainer.py -m gpu

# Megatron pre-PP readiness (Qwen3-VL MoE, 4-layer split)
HF_HOME=/mnt/local_storage/hf-cache \
pytest tests/megatron/test_engine_megatron_prepp.py -k test_megatron_engine_prepp -m gpu -v
```

Override Megatron sweep sizes with:
- `MEGATRON_SINGLE_TP_SIZE`, `MEGATRON_SINGLE_EP_SIZE`, `MEGATRON_SINGLE_NUM_ACTORS`
- `MEGATRON_TEST_TP_SIZES=2,4`, `MEGATRON_TEST_EP_SIZES=2,4`
- `MEGATRON_TEST_VISION_ACTORS`, `MEGATRON_TEST_TEXT_ACTORS`

### RDT Transport Test

```bash
pytest tests/parallel/test_rdt_non_collocated.py -m gpu -v
```

### Dataset Alignment Test

`tests/dataset/test_dataset_modalities.py::test_real_dataset_alignment` automatically skips if the COCO validation set is not present.

## Run All Tests

```bash
bash tests/run_all_tests.sh
```

## Tips

- **CPU tests**: Run with `pytest -m cpu_only`
- **GPU tests**: Run each file individually (do NOT run all together)
- Set `HF_HOME` if you want weights/datasets cached at a specific path
- For verbose logging during GPU runs, add `-s` and export `CUDA_LAUNCH_BLOCKING=1`
- Use `-v` flag for more detailed test output
