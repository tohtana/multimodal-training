# Pipeline Framework User Guide

This guide covers the flexible pipeline parallelism framework for training vision-language models (VLMs) with heterogeneous parallelism strategies across N stages.

## Table of Contents

1. [Overview](#1-overview)
2. [Terminology](#2-terminology)
3. [Architecture Overview](#3-architecture-overview)
4. [How to Define Stages](#4-how-to-define-stages)
5. [How to Define Resource Sets](#5-how-to-define-resource-sets)
6. [How to Define Edges and Placements](#6-how-to-define-edges-and-placements)
7. [How to Write a Config File](#7-how-to-write-a-config-file)
8. [How to Map a Model's Layers to Stages](#8-how-to-map-a-models-layers-to-stages)
9. [Adding Support for a New Model](#9-adding-support-for-a-new-model)
10. [Payload Interface](#10-payload-interface)
11. [Example Commands](#11-example-commands)
12. [Troubleshooting](#12-troubleshooting)
13. [Checkpoint Management](#13-checkpoint-management)
14. [Logging](#14-logging)
15. [Transport Tiers Deep Dive](#15-transport-tiers-deep-dive)
16. [Limitations](#16-limitations)
17. [Future Plans](#17-future-plans)

---

## 1. Overview

The pipeline framework enables **N-stage DAG-based training** with heterogeneous parallelism. Each stage in the pipeline (e.g., vision encoder, bridge adapter, text decoder) runs as an independent SPMD group with its own parallelism strategy and training engine. Stages are connected by directed edges that carry activations forward and gradients backward.

### When to use `train_pipeline.py` vs `train_ray.py`

| | `train_pipeline.py` | `train_ray.py` |
|---|---|---|
| **Entry point** | `python -m python.train_pipeline` | `python -m python.train_ray` |
| **Config** | Requires `pipeline:` section in YAML | Legacy two-stage format |
| **Stages** | N stages (2, 3, or more) | Exactly 2 (vision + text) |
| **GPU allocation** | Flexible per-stage resource sets | Same GPUs for all stages |
| **Use when** | 3+ stages, asymmetric GPU allocation, bridge adapters, MoE | Simple two-stage VLM training |

Both paths produce numerically equivalent results for two-stage VLM configurations (validated within 0.002 absolute loss across 3 iterations).

### Supported Use Cases

- **UC1**: Two-stage VLM (vision + text) on shared GPUs
- **UC2**: Three-stage pipeline with a learnable bridge adapter (vision → bridge → text)
- **UC3**: Asymmetric GPU allocation (e.g., 2 GPUs for vision, 4 GPUs for text)
- **UC4**: Interleaved MoE with overlapping GPU sets (attention TP + MoE EP)

---

## 2. Terminology

| Term | Definition |
|------|-----------|
| **Stage** | A model component (e.g., vision encoder, text decoder, bridge adapter) that runs as an independent SPMD group. |
| **Edge** | A directed data-flow connection between two stages; forward activations flow src→dst, gradients flow dst→src. |
| **Resource Set** | A named group of GPUs that stages can be placed on. |
| **Placement** | The mapping of a stage to a resource set. |
| **Collocation** | Multiple stages sharing the same physical GPUs via fractional GPU allocation. |
| **Transport Tier** | T0 (same-process zero-copy), T1 (same-GPU CUDA IPC), T2 (cross-GPU NCCL). |
| **Component Type** | The functional role of a stage: `vision`, `text`, or `bridge`. |
| **Engine** | The training backend: `native` (plain PyTorch), `deepspeed`, or `megatron`. |
| **Parallelism** | How a stage distributes work: `none`, `sequence`, `tensor`, `autotp`, or `expert`. |
| **ActorGroup** | A Ray actor collection forming one SPMD group with its own `torch.distributed` process group. |
| **Pipeline DAG** | Directed acyclic graph defining the stage topology and data flow. |
| **Microbatch** | A subdivision of the mini-batch for pipeline scheduling. |
| **1F1B** | "One Forward One Backward" schedule that interleaves forward and backward passes to reduce pipeline bubble time. |

---

## 3. Architecture Overview

```
                          ┌──────────────┐
                          │  Config YAML │
                          └──────┬───────┘
                                 │  parse_pipeline_dict()
                                 ▼
                          ┌──────────────┐
                          │ Pipeline DAG │  stages, edges, resource_sets, placements
                          └──────┬───────┘
                                 │  resolve_trainer() per stage
                                 ▼
                          ┌──────────────┐
                          │ ActorGroups  │  one per stage, each with N Ray actors
                          └──────┬───────┘
                                 │
                                 ▼
                ┌────────────────┼────────────────┐
                │                │                │
          ┌─────┴─────┐  ┌──────┴──────┐  ┌──────┴──────┐
          │  Vision    │  │   Bridge    │  │    Text     │
          │  Trainer   │  │   Trainer   │  │   Trainer   │
          │  (SP+ZeRO) │  │  (native)   │  │  (AutoTP)   │
          └────────────┘  └─────────────┘  └─────────────┘
```

### Entry Points

- **`python/train_pipeline.py`** — Pipeline path. Reads `pipeline:` section from YAML, builds DAG, creates ActorGroups per stage, runs `VLMPipelineRunner`.
- **`python/train_ray.py`** — Legacy path. Hardcoded two-stage vision+text logic.

### Key Source Files

| File | Purpose |
|------|---------|
| `python/pipeline/stage.py` | Core dataclasses: `Stage`, `ResourceSet`, `Placement`, `EdgeConfig`, `Pipeline` |
| `python/pipeline/dag.py` | `PipelineDAG` with topological sort, cycle detection, predecessor/successor queries |
| `python/pipeline/scheduler.py` | `SequentialScheduler` and `OneFOneBScheduler` (1F1B) |
| `python/pipeline/config_loader.py` | YAML → `Pipeline` parser with validation |
| `python/pipeline/vlm_runner.py` | `VLMPipelineRunner` — DAG-based orchestration of ActorGroups |
| `python/pipeline/placement.py` | `PlacementManager`, `PipelineActorGroup`, `StageModelSpec` |
| `python/pipeline/router.py` | `CrossStageRouter` + `RoutingPlan` for M:N actor mapping |
| `python/pipeline/ray_runner.py` | `RayPipelineRunner` — generic pipeline runner with microbatch tracking |
| `python/pipeline/layout.py` | `LayoutAdapter` framework (identity, shard, gather, sp_to_tp) |
| `python/ray/trainer.py` | Base `Trainer` class with abstract methods for all stages |
| `python/ray/vision.py` | `BaseVisionTrainer` and `QwenVisionTrainer` |
| `python/ray/text.py` | `BaseTextTrainer` and `QwenTextTrainer` |
| `python/ray/bridge.py` | `BridgeTrainer` — simplest stage implementation |
| `python/ray/payloads.py` | `StageOutputs`, `StageGradients`, `VisionOutputs`, `TextBackwardOutputs` |
| `python/ray/actor_group.py` | `ActorGroup` for SPMD groups with collocation support |
| `python/trainer_registry.py` | `register_trainer()` and `resolve_trainer()` |
| `python/train_pipeline.py` | Hydra entry point for pipeline training |
| `configs/pipeline_sample.yaml` | UC1 two-stage config |
| `configs/pipeline_bridge_sample.yaml` | UC2 three-stage config with bridge |

---

## 4. How to Define Stages

A stage represents one model component in the pipeline. Each stage is defined as a YAML dict under `pipeline.stages`.

### Stage Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | yes | Unique identifier for this stage |
| `component_type` | string | yes | Functional role: `vision`, `text`, or `bridge` |
| `is_source` | bool | no (default: false) | Whether this stage loads data (first stage in the DAG) |
| `is_terminal` | bool | no (default: false) | Whether this stage computes loss (last stage in the DAG) |
| `engine` | string | no | Overrides the component config's engine at pipeline level |
| `parallelism` | string | no | Overrides the component config's parallelism at pipeline level |

### Source and Terminal Stages

- **Source stage** (`is_source: true`): Owns the dataloader. Its `forward_step(iteration)` takes only the iteration number.
- **Terminal stage** (`is_terminal: true`): Computes the loss. Its `forward_step(upstream_ref, iteration)` returns a dict containing `loss`.
- **Intermediate stage**: Receives upstream activations and passes output to the next stage.

### Example

```yaml
pipeline:
  stages:
    - name: vision
      component_type: vision
      is_source: true       # loads images from dataloader
      is_terminal: false
    - name: bridge
      component_type: bridge
      is_source: false
      is_terminal: false     # intermediate — projects embeddings
    - name: text
      component_type: text
      is_source: false
      is_terminal: true      # computes language modeling loss
```

### How `component_type` Maps to Trainer Classes

The `component_type` and `engine` fields determine which trainer class is instantiated, via `trainer_registry.py`:

| component_type | engine | model_type | Trainer Class |
|---------------|--------|------------|---------------|
| `vision` | `deepspeed` | `qwen2_5_vl` | `QwenVisionTrainer` |
| `vision` | `native` | `qwen2_5_vl` | `QwenVisionTrainer` |
| `vision` | `megatron` | `qwen2_5_vl` | `MegatronVisionTrainer` |
| `text` | `deepspeed` | `qwen2_5_vl` | `QwenTextTrainer` |
| `text` | `native` | `qwen2_5_vl` | `QwenTextTrainer` |
| `text` | `megatron` | `qwen2_5_vl` | `MegatronTextTrainer` |
| `bridge` | `native` | any | `BridgeTrainer` |

---

## 5. How to Define Resource Sets

A resource set names a group of GPUs that stages can be placed on.

### Resource Set Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | yes | Unique identifier |
| `num_gpus` | int | yes | Number of GPUs in this set |
| `device_ids` | list[int] | no | Explicit GPU device IDs (e.g., `[0, 1, 2, 3]`) |
| `subset_of` | string | no | Parent resource set name (child devices must be a subset of parent) |

### Examples

**Shared GPUs** — all stages collocated on the same 4 GPUs:

```yaml
resource_sets:
  - name: shared_gpus
    num_gpus: 4
```

**Separate encoder/decoder GPUs** — different stages on different hardware:

```yaml
resource_sets:
  - name: encoder_gpus
    num_gpus: 2
  - name: decoder_gpus
    num_gpus: 8
```

**Overlapping GPU sets** — for MoE interleaving where attention uses a subset of MoE GPUs:

```yaml
resource_sets:
  - name: moe_gpus
    num_gpus: 8
    device_ids: [0, 1, 2, 3, 4, 5, 6, 7]
  - name: attn_gpus
    num_gpus: 4
    device_ids: [0, 1, 2, 3]
    subset_of: moe_gpus    # attn_gpus ⊆ moe_gpus
```

### Subset Rules

When using `subset_of`:
- Both the child and parent resource sets **must** have `device_ids` specified.
- The child's `device_ids` must be a proper subset of the parent's `device_ids`.
- Stages on a child resource set get fractional GPU allocation (e.g., 0.5 GPU per actor), enabling CUDA IPC (T1) transport between parent and child actors on the same physical GPU.

---

## 6. How to Define Edges and Placements

### Edges

An edge defines a directed data-flow connection between two stages.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `src` | string | yes | Source stage name |
| `dst` | string | yes | Destination stage name |
| `merge_policy` | string | no | For fan-in: `concat`, `sum`, or `dict` |

During forward: activations flow from `src` to `dst`.
During backward: gradients flow from `dst` back to `src`.

```yaml
edges:
  - src: vision
    dst: bridge
  - src: bridge
    dst: text
```

**Fan-in rule**: If a stage receives edges from multiple upstream stages, all incoming edges to that stage must specify a `merge_policy`:

```yaml
edges:
  - src: encoder_a
    dst: decoder
    merge_policy: concat
  - src: encoder_b
    dst: decoder
    merge_policy: concat
```

### Placements

A placement maps a stage to a resource set. Every stage must have exactly one placement.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `stage` | string | yes | Stage name |
| `resource_set` | string | yes | Resource set name |

```yaml
placements:
  - stage: vision
    resource_set: shared_gpus
  - stage: bridge
    resource_set: shared_gpus
  - stage: text
    resource_set: shared_gpus
```

---

## 7. How to Write a Config File

A pipeline config has two parts: the **pipeline structure** (stages, edges, resources, placements) and the **component configs** (model, training, data, deepspeed parameters).

### Full Annotated Example

Below is `configs/pipeline_bridge_sample.yaml` — a three-stage pipeline with a bridge adapter:

```yaml
# ── Pipeline Structure ──────────────────────────────────────────

pipeline:
  stages:
    - name: vision                 # Stage 1: vision encoder
      component_type: vision
      is_source: true              # Owns the dataloader
      is_terminal: false
    - name: bridge                 # Stage 2: learnable MLP projection
      component_type: bridge
      is_source: false
      is_terminal: false
    - name: text                   # Stage 3: language model decoder
      component_type: text
      is_source: false
      is_terminal: true            # Computes loss
  edges:
    - src: vision
      dst: bridge                  # vision embeddings → bridge MLP
    - src: bridge
      dst: text                    # projected embeddings → text decoder
  resource_sets:
    - name: shared_gpus
      num_gpus: 4                  # All stages share 4 GPUs
  placements:
    - stage: vision
      resource_set: shared_gpus
    - stage: bridge
      resource_set: shared_gpus
    - stage: text
      resource_set: shared_gpus

# ── Vision Model Configuration ──────────────────────────────────

vision:
  model_type: "qwen2_5_vl"
  model_name: "Qwen/Qwen2.5-VL-32B-Instruct"
  engine: "deepspeed"              # Use DeepSpeed for gradient reduction
  parallelism: "sequence"          # Sequence parallelism for long vision sequences
  dtype: "bfloat16"
  attention_backend: "sdpa"        # Use SDPA (not flash_attention_2)
  activation_checkpointing: true
  autocast: true
  num_hidden_layers: 4             # Override for smoke testing (remove for full model)
  engine_config:
    zero_stage: 1                  # ZeRO stage 1 (optimizer state partitioning)
    sequence_parallel_size: 4      # SP group size = number of GPUs

# ── Bridge Configuration ─────────────────────────────────────────

bridge:
  model_type: "generic"            # No specific model architecture
  engine: "native"                 # Plain PyTorch (no DeepSpeed)
  parallelism: "none"              # No parallelism — single MLP per GPU
  dtype: "bfloat16"
  autocast: true
  bridge_input_dim: 5120           # Must match vision output hidden size
  bridge_output_dim: 5120          # Must match text input hidden size

# ── Text Model Configuration ─────────────────────────────────────

text:
  model_type: "qwen2_5_vl"
  model_name: "Qwen/Qwen2.5-VL-32B-Instruct"
  engine: "deepspeed"
  parallelism: "autotp"            # DeepSpeed AutoTP for tensor parallelism
  dtype: "bfloat16"
  attention_backend: "sdpa"
  activation_checkpointing: true
  autocast: true
  num_hidden_layers: 4             # Override for smoke testing
  engine_config:
    zero_stage: 1
    autotp_size: 4                 # TP group size = number of GPUs
    tp_overlap_comm: false

# ── Training Configuration ────────────────────────────────────────

training:
  batch_size: 1
  learning_rate: 5e-05
  num_epochs: 1
  num_iterations: 3
  warmup_steps: 0
  warmup_ratio: 0.03
  lr_scheduler_type: "cosine"
  weight_decay: 0.01
  gradient_accumulation_steps: 1
  seed: 42
  parallel_size: 4                 # Number of actors per stage
  dp_size: 1                       # Data parallel size (must be 1)
  collocate: true                  # All stages share GPUs
  clip_grad_norm: false
  max_grad_norm: 1.0
  no_checkpoint: true
  checkpoint_dir: "/tmp/pipeline_bridge_checkpoints"
  log_interval: 1

# ── Data Configuration ────────────────────────────────────────────

data:
  datasets:
    - laion_pop_train
  data_registry:
    laion_pop_train:
      annotation_path: "/mnt/local_storage/laion/laion_pop_train.jsonl"
      data_path: "/mnt/local_storage/laion/images"
  num_workers: 4
  pin_memory: true
  data_flatten: true
  min_pixels: 802816
  max_pixels: 802816
  force_fixed_size: false

# ── DeepSpeed Configuration ───────────────────────────────────────

deepspeed:
  reduce_bucket_size: 500000000    # 500M elements per all-reduce bucket
```

### Key Config Parameters

| Section | Parameter | Description | Default |
|---------|-----------|-------------|---------|
| `training` | `parallel_size` | Number of actors per stage (SP/TP group size) | required |
| `training` | `dp_size` | Data parallel size (must be 1 for now) | 1 |
| `training` | `collocate` | Whether stages share GPUs | required |
| `training` | `batch_size` | Micro-batch size per actor | required |
| `training` | `num_iterations` | Training steps per epoch | required |
| `training` | `clip_grad_norm` | Enable global gradient norm clipping | false |
| `vision` | `parallelism` | `sequence`, `tensor`, `deepspeed`, or `none` | required |
| `vision` | `engine` | `deepspeed`, `native`, or `megatron` | required |
| `vision` | `num_hidden_layers` | Override vision encoder depth (for testing) | full model |
| `text` | `parallelism` | `autotp`, `tensor`, `deepspeed`, or `none` | required |
| `text` | `engine` | `deepspeed`, `native`, or `megatron` | required |
| `text` | `num_hidden_layers` | Override decoder depth (for testing) | full model |
| `bridge` | `bridge_input_dim` | Input dimension (must match upstream output) | 5120 |
| `bridge` | `bridge_output_dim` | Output dimension (must match downstream input) | 5120 |
| `deepspeed` | `reduce_bucket_size` | All-reduce bucket size in elements | required |

### Engine-Specific Config Keys

Engine-specific parameters go under the `engine_config` sub-dict of each component:

| Engine | Key | Description |
|--------|-----|-------------|
| `deepspeed` | `zero_stage` | ZeRO optimization stage (0, 1, 2) |
| `deepspeed` | `sequence_parallel_size` | SP group size (for vision with SP) |
| `deepspeed` | `autotp_size` | AutoTP group size (for text with TP) |
| `deepspeed` | `tp_overlap_comm` | Overlap TP communication with compute |
| `deepspeed` | `reduce_bucket_size` | Per-component bucket size override |

---

## 8. How to Map a Model's Layers to Stages

### Step-by-Step Process

1. **Identify the model's major components.** For a VLM, these are typically: vision encoder, optional projection/bridge, and language model decoder.

2. **Decide parallelism strategy per component:**
   - Vision encoder: typically has long sequences (up to 65k tokens from high-res images) but fewer parameters (~670M). Use **Sequence Parallelism (SP)** with ZeRO-1.
   - Text decoder: large parameter count (7B–72B) but shorter sequences. Use **Tensor Parallelism (AutoTP)** via DeepSpeed.
   - Bridge adapter: small parameter count, no parallelism needed. Use **native** engine.

3. **Map each component to a `component_type`:**
   - Vision encoder → `vision`
   - Projection layer / bridge → `bridge`
   - Language model → `text`

4. **Choose engine per component:**
   - SP + ZeRO-1 → `deepspeed`
   - AutoTP → `deepspeed`
   - Plain PyTorch → `native`
   - Megatron-LM → `megatron`

5. **Define resource allocation:**
   - How many GPUs per component?
   - Are components collocated (shared GPUs) or on separate GPUs?

### Worked Example: Qwen2.5-VL-32B

| Component | Params | Sequence Length | Parallelism | Engine | GPUs |
|-----------|--------|----------------|-------------|--------|------|
| Vision encoder | ~670M | Up to 65k tokens | SP | deepspeed (ZeRO-1) | 4 |
| Bridge MLP | ~50M | Same as vision out | none | native | 4 (collocated) |
| Text decoder | ~32B | ~2k tokens | AutoTP | deepspeed (ZeRO-1) | 4 (collocated) |

This maps to:
```yaml
pipeline:
  stages:
    - name: vision
      component_type: vision
      is_source: true
    - name: bridge
      component_type: bridge
    - name: text
      component_type: text
      is_terminal: true
  edges:
    - src: vision
      dst: bridge
    - src: bridge
      dst: text
  resource_sets:
    - name: shared_gpus
      num_gpus: 4
  placements:
    - stage: vision
      resource_set: shared_gpus
    - stage: bridge
      resource_set: shared_gpus
    - stage: text
      resource_set: shared_gpus
```

---

## 9. Adding Support for a New Model

To add a new model to the pipeline framework, you need to implement trainer subclasses for each component type and register them.

### Vision Component

Subclass `BaseVisionTrainer` (defined in `python/ray/vision.py`) and implement these 7 abstract methods:

```python
class MyVisionTrainer(BaseVisionTrainer):

    def _create_model_instance(self, model_config):
        """Create model and optional projector instances.
        Returns: (model, projector) tuple where projector can be None."""

    def _get_projector_or_merger(self, model, projector):
        """Get the projector or merger module for parallelization.
        Returns: module to parallelize (or None)."""

    def _parallelize_projector_or_merger(self, model, projector, tp_mesh):
        """Apply tensor parallelism to projector/merger."""

    def _setup_sequence_parallel(self, model, sp_group):
        """Set up sequence parallelism group assignment.
        Assign sp_group to the appropriate model submodules."""

    def _get_vision_config(self, model_name):
        """Get vision config for dataset creation.
        Returns: vision configuration object."""

    def _model_forward(self, batch):
        """Forward pass through model.
        Args: batch dict with 'pixel_values' and 'image_grid_thw'.
        Returns: vision output tensor [batch_size, num_tokens, hidden_size]."""

    def _zero_padded_weights_after_init(self, model, projector):
        """Zero out padded weights after init (for TP with meta device).
        No-op if your model doesn't use padded attention heads."""
```

You also need to implement the 5 abstract methods from the base `Trainer` class:

```python
    def _load_model_config(self, model_name):
        """Load and return model config from pretrained path."""

    def _get_transformer_layers(self, model):
        """Get the list of transformer layers for TP parallelization."""

    def _get_tensor_parallel_mapping(self):
        """Get dict mapping layer names to ColwiseParallel/RowwiseParallel."""

    def save_checkpoint(self, checkpoint_dir, epoch):
        """Save model checkpoint. Returns: path to saved checkpoint."""

    def load_checkpoint(self, checkpoint_dir, epoch):
        """Load model checkpoint. Returns: True if successful."""
```

See `QwenVisionTrainer` in `python/ray/vision.py:668` for a full implementation example.

### Text Component

Subclass `BaseTextTrainer` (defined in `python/ray/text.py`) and implement these 2 abstract methods:

```python
class MyTextTrainer(BaseTextTrainer):

    def _create_model_and_lm_head(self, model_config):
        """Create model and lm_head instances.
        Returns: (model, lm_head) tuple."""

    def _get_embedding_module(self, model):
        """Get the embedding module from the model.
        Returns: embedding module (e.g., model.embed_tokens)."""
```

Plus the same 5 `Trainer` abstract methods (`_load_model_config`, `_get_transformer_layers`, `_get_tensor_parallel_mapping`, `save_checkpoint`, `load_checkpoint`).

See `QwenTextTrainer` in `python/ray/text.py` for a full implementation example.

### Bridge / Adapter Component

Subclass `Trainer` directly (defined in `python/ray/trainer.py`). This is the simplest case. Implement:

```python
class MyBridgeTrainer(Trainer):

    def build_model(self):
        """Create the bridge model, optimizer, and scheduler."""

    def forward_step(self, upstream_ref, iteration=-1):
        """Receive upstream payload, project, return VisionOutputs."""

    def backward_step(self, downstream_grad_ref):
        """Receive downstream gradient, backprop, return TextBackwardOutputs."""

    # Required Trainer abstract methods:
    def _load_model_config(self, model_name):
        return None  # Bridge typically doesn't need model config

    def _get_transformer_layers(self, model):
        return []    # No transformer layers to parallelize

    def _get_tensor_parallel_mapping(self):
        return {}    # No TP mapping

    def save_checkpoint(self, checkpoint_dir, epoch):
        """Save bridge weights."""

    def load_checkpoint(self, checkpoint_dir, epoch):
        """Load bridge weights."""
```

See `BridgeTrainer` in `python/ray/bridge.py` for a complete, minimal example.

### Register in `trainer_registry.py`

After implementing your trainer class, register it so the pipeline framework can resolve it from config:

```python
from python.trainer_registry import register_trainer
from my_module import MyVisionTrainer, MyTextTrainer

register_trainer(
    component_type="vision",
    engine="deepspeed",
    model_type="my_model",
    trainer_cls=MyVisionTrainer,
)

register_trainer(
    component_type="text",
    engine="deepspeed",
    model_type="my_model",
    trainer_cls=MyTextTrainer,
)
```

Alternatively, add your model to the `_resolve_default_trainer()` function in `python/trainer_registry.py` for automatic resolution.

---

## 10. Payload Interface

Stages communicate via typed payload dataclasses defined in `python/ray/payloads.py`.

### Forward Payloads

| Class | Fields | Used By |
|-------|--------|---------|
| `StageOutputs` | `activations`, `attention_mask`, `meta` | Generic pipeline stages |
| `VisionOutputs` | `embeddings`, `attention_mask`, `meta` | Legacy vision → text (backward-compatible alias: `activations` ↔ `embeddings`) |

### Backward Payloads

| Class | Fields | Used By |
|-------|--------|---------|
| `StageGradients` | `grad`, `meta` | Generic pipeline stages |
| `TextBackwardOutputs` | `grad`, `meta` | Legacy text → vision (structurally identical to `StageGradients`) |

### Normalization Functions

These functions convert between payload types:

```python
normalize_vision_outputs(payload)     # dict/StageOutputs/VisionOutputs → VisionOutputs
normalize_stage_outputs(payload)      # dict/VisionOutputs/StageOutputs  → StageOutputs
normalize_text_backward_outputs(payload)  # dict/StageGradients/TextBackwardOutputs → TextBackwardOutputs
normalize_stage_gradients(payload)    # dict/TextBackwardOutputs/StageGradients → StageGradients
```

### Data Flow Rule

- **Source stage** `forward_step(iteration)` → returns `VisionOutputs` (or `StageOutputs`)
- **Intermediate stage** `forward_step(upstream_ref, iteration)` → receives upstream payload, returns `VisionOutputs`
- **Terminal stage** `forward_step(upstream_ref, iteration)` → receives upstream payload, computes loss, returns dict with `"loss"` key
- **Backward** flows in reverse: terminal `backward_step()` → intermediate `backward_step(grad_ref)` → source `backward_step(grad_ref)`

---

## 11. Example Commands

### UC1: Two-Stage VLM

```bash
cd multimodal-training
python -m python.train_pipeline --config-path=../configs --config-name=pipeline_sample
```

### UC2: Three-Stage with Bridge

```bash
cd multimodal-training
python -m python.train_pipeline --config-path=../configs --config-name=pipeline_bridge_sample
```

### UC3: Asymmetric GPU Allocation (test only)

```bash
cd multimodal-training
python -m pytest tests/test_pipeline_asymmetric.py -m gpu -v
```

### UC4: Interleaved MoE (test only, Python API)

```bash
cd multimodal-training
python -m pytest tests/test_pipeline_moe_interleaved.py -m gpu -v
```

### Running Tests

```bash
# All CPU tests (fast, no GPU required)
cd multimodal-training && pytest -m cpu_only

# Full test suite (CPU + GPU)
cd multimodal-training && bash tests/run_all_tests.sh

# Pipeline-specific CPU tests
cd multimodal-training && pytest tests/test_pipeline_dag.py -m cpu_only -v
cd multimodal-training && pytest tests/test_pipeline_payload_contract.py -m cpu_only -v

# Pipeline GPU tests (run individually)
cd multimodal-training && pytest tests/test_pipeline_vlm.py -m gpu -v
cd multimodal-training && pytest tests/test_pipeline_three_stage.py -m gpu -v
cd multimodal-training && pytest tests/test_pipeline_1f1b.py -m gpu -v
cd multimodal-training && pytest tests/test_pipeline_overlap.py -m gpu -v
```

### Hydra Overrides

Override any config value from the command line:

```bash
python -m python.train_pipeline --config-path=../configs --config-name=pipeline_sample \
    training.batch_size=2 \
    training.num_iterations=10 \
    training.collocate=false \
    vision.num_hidden_layers=8
```

---

## 12. Troubleshooting

### Common Errors

**GPU resource exhaustion / OOM**
- Reduce `num_hidden_layers` for smoke testing (e.g., 4 instead of 64).
- Full 32B models with 64 layers require ZeRO-2 or more GPUs when collocated.
- Reduce `batch_size` to 1.
- Enable `activation_checkpointing: true`.

**NCCL timeout during actor initialization**
- Ensure all actors can reach each other on the network.
- Check that `parallel_size` matches `num_gpus` in the resource set for collocated configs.
- Increase NCCL timeout: `export NCCL_TIMEOUT=1800`

**Placement group creation failure**
- Ray placement groups need enough CPU resources for all collocated actors. Each GPU bundle needs CPU allocations for all collocated stages, not just one.
- Check `ray status` to verify available resources.

**`flash_attention_2` ABI mismatch**
- Use `attention_backend: "sdpa"` instead of `"flash_attention_2"`. Flash attention has ABI compatibility issues between Ray runtime's flash_attn and the training environment's PyTorch.

**Config validation error: "dp_size must be 1"**
- Global data parallelism across the pipeline is not yet supported. Set `dp_size: 1`.

**"Pipeline must have at least one source stage"**
- Ensure exactly one stage has `is_source: true` in your config.
- Ensure exactly one stage has `is_terminal: true`.

**Dimension mismatch in backward pass**
- Check that `bridge_input_dim` / `bridge_output_dim` match the upstream and downstream hidden sizes.
- Vision outputs are always 3D `[batch_size, num_tokens, hidden_size]`; gradients may be squeezed. The framework handles dimension matching automatically.

---

## 13. Checkpoint Management

### How Checkpoints Work

Each stage saves its own checkpoint independently. The pipeline runner coordinates saving and loading across all stages.

**Save** (`VLMPipelineRunner.save_checkpoint`):
- Each stage's `save_checkpoint(checkpoint_dir, epoch)` is called.
- Checkpoints are stored under `{checkpoint_dir}/epoch_{epoch}/{component}/rank_{rank}.pt`.
- A `metadata.json` file records the epoch, timestamp, stage names, and per-stage checkpoint paths.

**Load** (`VLMPipelineRunner.load_checkpoint`):
- Each stage's `load_checkpoint(checkpoint_dir, epoch)` is called.
- Returns `True` only if all stages loaded successfully.

### Auto-Resume

When `no_checkpoint: false` and `checkpoint_dir` is set, the training script automatically:
1. Checks for the latest checkpoint via `find_latest_checkpoint(checkpoint_dir)`.
2. If found, loads all stage checkpoints and resumes from the next epoch.

### Checkpoint Directory Structure

```
checkpoint_dir/
├── epoch_0/
│   ├── metadata.json
│   ├── vision/
│   │   ├── rank_0.pt
│   │   ├── rank_1.pt
│   │   └── ...
│   ├── bridge/
│   │   └── rank_0.pt
│   └── text/
│       ├── rank_0.pt
│       └── ...
└── epoch_1/
    └── ...
```

---

## 14. Logging

### Log Location

Training logs are written to `logs/archive/` with timestamped filenames. The `RAY_TRAIN_LOG_FILE` environment variable controls the log file path.

### Per-Actor Logs

Each Ray actor logs with a `[r{rank}]` prefix, making it easy to filter:

```
[r0] Vision forward_step: iteration=0, sample_index=42
[r1] Vision forward_step: iteration=0, sample_index=42
[r0] QwenTextTrainer forward_step: iteration=0
```

### Debug Logging

Enable debug logging to see detailed model building, parallelism setup, and training step timing:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Or set `profile_time: true` in the config to get per-step timing in the log output.

---

## 15. Transport Tiers Deep Dive

The framework selects transport automatically based on where source and destination actors are placed.

### T0: Same-Process Zero-Copy

- **When**: Source and destination stages are collocated on the same GPU, and the data is a Ray ObjectRef passed between actors on the same node.
- **Performance**: Near-zero overhead. Tensors are passed by reference via Ray's object store.
- **Used by**: Collocated vision→text or vision→bridge→text on shared GPUs.

### T1: Same-GPU CUDA IPC

- **When**: Source and destination actors are on the same physical GPU but in different processes (overlapping resource sets with `subset_of`).
- **Performance**: Fast GPU-to-GPU transfer without going through CPU. Requires `.detach()` before handle creation (autograd graphs don't cross process boundaries).
- **Used by**: Interleaved MoE where attention (TP) and MoE FFN (EP) stages share GPUs.

### T2: Cross-GPU NCCL

- **When**: Source and destination actors are on different physical GPUs.
- **Performance**: Uses NCCL for efficient cross-GPU communication. Applied per-call with `.options(tensor_transport="nccl")` (not as a decorator).
- **Used by**: Non-collocated stages on separate GPU sets.

### How Collocation Affects Transport

| Configuration | Transport | Notes |
|--------------|-----------|-------|
| `collocate: true`, same resource set | T0 | Zero-copy via Ray ObjectRef |
| `subset_of` overlapping GPUs | T1 | CUDA IPC between processes on same GPU |
| Separate resource sets, different GPUs | T2 | NCCL cross-GPU |
| Bridge stage (any config) | Ray ObjectRef | Bridge skips IPC, uses ObjectRef passing |

---

## 16. Limitations

- **`dp_size` must be 1**: No global data parallelism across the pipeline yet.
- **Gradient accumulation**: `gradient_accumulation_steps` must be ≤ 1 when using microbatches.
- **Single source and terminal**: The pipeline requires exactly one source stage and one terminal stage.
- **Bridge engine**: Bridge only supports `engine: native` (no DeepSpeed or Megatron bridge yet).
- **Layout adapters**: Local tensor transforms only — no distributed all-gather/scatter.
- **MoE TP↔EP adapters**: Use identity per-actor; real expert routing needs distributed all-to-all.
- **Attention backend**: SDPA required. `flash_attention_2` has ABI mismatch with Ray runtime.
- **Full 32B models**: 64 layers require ZeRO-2 or more GPUs when collocated on 4×80GB GPUs.
- **MoE YAML config**: Interleaved MoE topologies are not configurable via YAML — use the Python API (`moe_config_gen.py`).

---

## 17. Future Plans

- **Global data parallelism** (`dp_size > 1`) across the pipeline.
- **Distributed layout adapters** with all-gather/scatter/all-to-all for real expert parallelism.
- **Full model testing** with ZeRO-2+ on larger GPU clusters (8+ GPUs).
- **CUDA MPS** for compute overlap on shared GPUs (MoE interleaving).
- **Fan-out/fan-in DAG topologies** beyond linear pipelines.
- **Bridge engine backends** (DeepSpeed, Megatron) for distributed bridge training.
- **Performance benchmarking**: pipeline bubble ratio, throughput metrics, 1F1B vs sequential comparison.
- **Remove legacy custom model code** (`python/models/qwen2_5_vl/`) — trainers now use HuggingFace native.
