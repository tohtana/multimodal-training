# Flexible Pipeline Parallelism
#
# This package implements N-stage heterogeneous pipeline parallelism for
# training vision-language models with per-stage GPU allocation, parallelism
# strategy, and backend engine.

from .dag import PipelineDAG
from .stage import EdgeConfig, Pipeline, Placement, ResourceSet, Stage

__all__ = [
    "Stage",
    "ResourceSet",
    "Placement",
    "EdgeConfig",
    "Pipeline",
    "PipelineDAG",
]
