"""Pipeline DAG: topological sort, cycle detection, connectivity checks (design §4.2, §6.1)."""

from __future__ import annotations

from collections import defaultdict, deque

from .stage import Pipeline


class PipelineDAGError(Exception):
    """Raised when the pipeline DAG is invalid."""


class PipelineDAG:
    """Directed acyclic graph for pipeline stages."""

    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline
        self._adj: dict[str, list[str]] = defaultdict(list)
        self._in_degree: dict[str, int] = {}
        self._reverse_adj: dict[str, list[str]] = defaultdict(list)

        stage_names = {s.name for s in pipeline.stages}
        for name in stage_names:
            self._in_degree[name] = 0

        for edge in pipeline.edges:
            self._adj[edge.src].append(edge.dst)
            self._reverse_adj[edge.dst].append(edge.src)
            self._in_degree[edge.dst] = self._in_degree.get(edge.dst, 0) + 1

    def topological_sort(self) -> list[str]:
        """Return stages in topological order. Raises PipelineDAGError on cycles."""
        in_degree = dict(self._in_degree)
        queue = deque([name for name, deg in in_degree.items() if deg == 0])
        order = []

        while queue:
            # Sort to ensure deterministic order among nodes with same in-degree
            queue = deque(sorted(queue))
            node = queue.popleft()
            order.append(node)
            for neighbor in sorted(self._adj.get(node, [])):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(order) != len(self._in_degree):
            visited = set(order)
            cycle_nodes = [n for n in self._in_degree if n not in visited]
            raise PipelineDAGError(f"Cycle detected involving stages: {cycle_nodes}")

        return order

    def is_connected(self) -> bool:
        """Check if the DAG is weakly connected (all stages reachable via undirected edges)."""
        if not self.pipeline.stages:
            return True

        # Build undirected adjacency
        undirected: dict[str, set[str]] = defaultdict(set)
        for edge in self.pipeline.edges:
            undirected[edge.src].add(edge.dst)
            undirected[edge.dst].add(edge.src)

        # BFS from first stage
        start = self.pipeline.stages[0].name
        visited = {start}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for neighbor in undirected.get(node, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return len(visited) == len(self.pipeline.stages)

    def sources(self) -> list[str]:
        """Return stage names with no incoming edges (DAG sources)."""
        return [name for name, deg in self._in_degree.items() if deg == 0]

    def sinks(self) -> list[str]:
        """Return stage names with no outgoing edges (DAG sinks)."""
        all_srcs = set()
        for neighbors in self._adj.values():
            all_srcs.update(neighbors)
        stage_names = {s.name for s in self.pipeline.stages}
        return [name for name in stage_names if name not in self._adj or not self._adj[name]]

    def predecessors(self, stage_name: str) -> list[str]:
        """Return the direct predecessors of a stage."""
        return list(self._reverse_adj.get(stage_name, []))

    def successors(self, stage_name: str) -> list[str]:
        """Return the direct successors of a stage."""
        return list(self._adj.get(stage_name, []))

    def validate(self) -> list[str]:
        """Full DAG validation. Returns list of error messages (empty = valid)."""
        errors = []

        # Pipeline-level validation
        errors.extend(self.pipeline.validate())

        # Cycle detection
        try:
            self.topological_sort()
        except PipelineDAGError as e:
            errors.append(str(e))

        # Connectivity check
        if not self.is_connected():
            errors.append("Pipeline DAG is not connected: some stages are unreachable")

        # All stages must be reachable from at least one source
        sources = self.sources()
        if not sources:
            errors.append("Pipeline has no source stages (stages with no incoming edges)")

        # Sinks must exist (at least one terminal)
        sinks = self.sinks()
        if not sinks:
            errors.append("Pipeline has no sink stages (stages with no outgoing edges)")

        return errors
