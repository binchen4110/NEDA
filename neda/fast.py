from collections import OrderedDict, defaultdict
from dataclasses import dataclass, fields

import numpy as np

from .data import Graph, History, Patch


def integers(values):
    return np.asarray(values, dtype=np.int64)


@dataclass
class Chunk:
    nodes: np.ndarray
    source: np.ndarray
    destination: np.ndarray
    relation: np.ndarray
    centers: np.ndarray
    override_positions: np.ndarray
    override_values: np.ndarray


def pack_graphs(frames, node_budget, edge_budget):

    chunks = []
    arrays = [[] for _ in range(7)]
    size, edges = 0, 0

    def finish():
        if size:
            chunks.append(Chunk(*(np.concatenate(a) if a else integers([]) for a in arrays)))

    for graph, override_pos, override_idx in frames:
        if size and (
            size + len(graph.nodes) > node_budget or edges + len(graph.source) > edge_budget
        ):
            finish()
            arrays = [[] for _ in range(7)]
            size, edges = 0, 0
        arrays[0].append(graph.nodes)
        arrays[1].append(graph.source + size)
        arrays[2].append(graph.destination + size)
        arrays[3].append(graph.relation)
        arrays[4].append(graph.center_indices + size)
        arrays[5].append(integers(override_pos) + size)
        arrays[6].append(integers(override_idx))
        size += len(graph.nodes)
        edges += len(graph.source)
    finish()
    return chunks


@dataclass
class Plan:
    baseline: list
    explicit: list
    occurrence_nodes: np.ndarray
    occurrence_entities: np.ndarray
    implicit_entities: np.ndarray
    implicit_begin: np.ndarray
    implicit_end: np.ndarray
    implicit_last: np.ndarray
    implicit_counts: np.ndarray
    frame_baseline: np.ndarray
    modified_frames: np.ndarray
    frame_groups: np.ndarray
    patch_last: np.ndarray
    patch_counts: np.ndarray
    patch_roots: np.ndarray
    patch_positions: np.ndarray
    roots: np.ndarray
    recent_gru: object = None

    @property
    def nbytes(self):
        total = 0
        for field in fields(self):
            value = getattr(self, field.name)
            if value is None:
                continue
            if hasattr(value, "nbytes"):
                total += value.nbytes
            else:
                total += sum(
                    getattr(chunk, f.name).nbytes
                    for chunk in value
                    if not getattr(chunk, "_shared", False)
                    for f in fields(chunk)
                )
        return total


class FastHistory(History):
    def __init__(self, *args, plan_cache_bytes=4 * 1024**3, **kwargs):
        super().__init__(*args, **kwargs)
        self.plan_cache_bytes = plan_cache_bytes
        self._plans = OrderedDict()
        self._plan_bytes = 0
        self.plan_builds, self.plan_hits = 0, 0
        empty = integers([])
        self._snapshot_graphs = [Graph(np.arange(self.num_entities), empty, empty, empty, empty)]

        self._lookup = np.tile(np.arange(self.num_entities), (len(self.snapshots), 1))
        offset = self.num_entities
        first_seen = defaultdict(dict)
        for i, snapshot in enumerate(self.snapshots):
            graph = snapshot.graph()
            self._snapshot_graphs.append(graph)
            self._lookup[i, graph.nodes] = np.arange(offset, offset + len(graph.nodes))
            offset += len(graph.nodes)
            for s, outgoing in snapshot.adjacency.items():
                for o, r in outgoing:
                    first_seen[s, r].setdefault(o, i)
        self._first_seen = {
            key: (integers(list(values)), integers(list(values.values())))
            for key, values in first_seen.items()
        }
        self._baseline_pools = {}
        self._relation_metadata = OrderedDict()
        self._relation_metadata_bytes = 0
        self._relation_metadata_budget = min(plan_cache_bytes // 8, 256 * 1024**2)
        self._snapshot_offsets = np.r_[
            self.num_entities,
            self.num_entities + np.cumsum([len(g.nodes) for g in self._snapshot_graphs[1:]]),
        ]
        self._explicit_runs = []
        for times in self.occurrences:
            t = integers(times)
            breaks = np.flatnonzero(np.diff(t) != 1) + 1
            starts = np.r_[0, breaks] if len(t) else integers([])
            stops = np.r_[breaks, len(t)] if len(t) else integers([])
            self._explicit_runs.append((t[starts], t[stops - 1] + 1))

    # 论文 §3.1 Patch Construction — Explicit Patch Division。
    def explicit_patches(self, root, cutoff):

        if not 0 <= cutoff <= len(self.snapshots):
            raise ValueError("Invalid history cutoff")
        starts, stops = self._explicit_runs[root]
        end = int(np.searchsorted(starts, cutoff))
        return [
            Patch(True, int(a), min(int(b), cutoff), (root,))
            for a, b in zip(starts[:end], stops[:end])
        ]

    def baseline_chunks(self, cutoff, node_budget, edge_budget):

        key = node_budget, edge_budget
        if key not in self._baseline_pools:
            chunks = pack_graphs(((g, (), ()) for g in self._snapshot_graphs), *key)
            ends, size, edges = [], 0, 0
            for i, graph in enumerate(self._snapshot_graphs):
                if size and (
                    size + len(graph.nodes) > node_budget or edges + len(graph.source) > edge_budget
                ):
                    ends.append(i)
                    size, edges = 0, 0
                size += len(graph.nodes)
                edges += len(graph.source)
            if size:
                ends.append(len(self._snapshot_graphs))
            assert len(chunks) == len(ends)
            for chunk in chunks:
                chunk._shared = True
            self._baseline_pools[key] = chunks, integers(ends)
        chunks, ends = self._baseline_pools[key]
        stop = cutoff + 1
        complete = int(np.searchsorted(ends, stop, side="right"))
        result = chunks[:complete]
        begin = int(ends[complete - 1]) if complete else 0
        if begin < stop:
            result = result + pack_graphs(
                ((g, (), ()) for g in self._snapshot_graphs[begin:stop]), *key
            )
        return result

    def relation_frame(self, time, num_relations):

        key = (time, num_relations)
        if key in self._relation_metadata:
            self._relation_metadata.move_to_end(key)
            return self._relation_metadata[key]
        graph = self._snapshot_graphs[time + 1]
        s, r, o = self.snapshots[time].triples.T
        pairs = np.unique(
            np.concatenate((np.stack((r, s), -1), np.stack((r + num_relations, o), -1))), axis=0
        )
        relation = pairs[:, 0].copy()
        positions = np.searchsorted(graph.nodes, pairs[:, 1])
        if not np.array_equal(graph.nodes[positions], pairs[:, 1]):
            raise ValueError("Relation participant is absent from snapshot graph")
        degree = np.bincount(relation, minlength=2 * num_relations).clip(min=1)
        result = relation, positions, degree
        size = sum(x.nbytes for x in result)
        if size <= self._relation_metadata_budget:
            while (
                self._relation_metadata
                and self._relation_metadata_bytes + size > self._relation_metadata_budget
            ):
                _, old = self._relation_metadata.popitem(last=False)
                self._relation_metadata_bytes -= sum(x.nbytes for x in old)
            self._relation_metadata[key] = result
            self._relation_metadata_bytes += size
        return result

    def neighbor_relation_time_edges(self, cutoff, heads, history_len=None):

        if not 0 <= cutoff < len(self.snapshots):
            raise ValueError("Relation/time neighbors require a known query timestamp")
        window = self.neighbor_history_window(cutoff, history_len)
        heads = np.unique(np.asarray(heads, dtype=np.int64))
        if len(heads) and (heads[0] < 0 or heads[-1] >= self.num_entities):
            raise ValueError("Invalid neighbor receiver")
        if not hasattr(self, "_typed_lookup"):
            lengths = np.fromiter((len(e) for e in self.events), dtype=np.int64)
            receivers = np.repeat(np.arange(self.num_entities), lengths)
            neighbors = np.fromiter(
                (e.neighbor for events in self.events for e in events), dtype=np.int64
            )
            types = np.fromiter(
                (e.relation for events in self.events for e in events), dtype=np.int64
            )
            times = np.concatenate(self._event_snapshots) if len(self.events) else integers([])
            order = np.lexsort((times, types, neighbors, receivers))
            receivers, neighbors, types, times = (
                v[order] for v in (receivers, neighbors, types, times)
            )
            boundaries = (
                np.r_[
                    True,
                    (np.diff(receivers) != 0) | (np.diff(neighbors) != 0) | (np.diff(types) != 0),
                ]
                if len(times)
                else np.empty(0, dtype=bool)
            )
            starts = np.flatnonzero(boundaries)
            group = np.cumsum(boundaries) - 1
            stride = len(self.snapshots) + 1
            encoded = group * stride + times
            self._typed_lookup = (
                receivers[starts],
                neighbors[starts],
                types[starts],
                starts,
                times,
                encoded,
                stride,
            )
            self._snapshot_times = np.asarray([s.time for s in self.snapshots], dtype=np.int64)
        rows, columns, types, starts, times, encoded, stride = self._typed_lookup
        mask = np.zeros(self.num_entities, dtype=bool)
        mask[heads] = True
        selected = np.flatnonzero(mask[rows])
        end = np.searchsorted(encoded, selected * stride + cutoff, side="left") - 1
        present = end >= starts[selected]
        selected, end = selected[present], end[present]
        previous = times[end]
        inside = previous >= window.start
        selected, previous = selected[inside], previous[inside]
        ages = (self.snapshots[cutoff].time - self._snapshot_times[previous]) / self.time_unit
        return rows[selected], columns[selected], types[selected], ages

    def query_vocabulary(self, cutoff):

        if not 0 <= cutoff <= len(self.snapshots):
            raise ValueError("Invalid history cutoff")
        from .vocabulary import QueryPastVocabulary

        return QueryPastVocabulary(self._first_seen, cutoff)

    def vocabulary(self, cutoff):

        result = defaultdict(set)
        for key, (objects, first_times) in self._first_seen.items():
            known = objects[first_times < cutoff]
            if len(known):
                result[key] = set(known.tolist())
        return result

    def execution_plan(self, cutoff, roots, implicit, node_budget, edge_budget):
        if implicit:
            raise ValueError("This NEDA version encodes explicit patches only")
        key = (cutoff, tuple(roots), implicit, node_budget, edge_budget)
        if key in self._plans:
            self._plans.move_to_end(key)
            self.plan_hits += 1
            return self._plans[key]
        from .preprocess import read_plan

        plan = read_plan(self, key)
        if plan is None:
            plan = self._build_plan(cutoff, roots, implicit, node_budget, edge_budget)
            from .patch_gru import RecentGRUPlan

            plan.recent_gru = RecentGRUPlan.build(plan)
            self.plan_builds += 1
        if plan.nbytes <= self.plan_cache_bytes:
            while self._plans and self._plan_bytes + plan.nbytes > self.plan_cache_bytes:
                _, old = self._plans.popitem(last=False)
                self._plan_bytes -= old.nbytes
            self._plans[key] = plan
            self._plan_bytes += plan.nbytes
        return plan

    def _build_plan(self, cutoff, roots, implicit, node_budget, edge_budget):
        if implicit:
            raise ValueError("This NEDA version encodes explicit patches only")
        baseline = self.baseline_chunks(cutoff, node_budget, edge_budget)
        frame_baseline, frame_groups = [], []
        patch_last, patch_counts, patch_roots, patch_positions = [], [], [], []
        for root_id, root in enumerate(roots):
            for position, patch in enumerate(self.explicit_patches(root, cutoff)):
                group = len(patch_counts)
                for time in range(patch.start, patch.stop):
                    frame_baseline.append(self._lookup[time, root])
                    frame_groups.append(group)
                patch_last.append(len(frame_baseline) - 1)
                patch_counts.append(patch.stop - patch.start)
                patch_roots.append(root_id)
                patch_positions.append(position)
        empty = integers([])
        return Plan(
            baseline=baseline,
            explicit=[],
            occurrence_nodes=empty,
            occurrence_entities=empty,
            implicit_entities=empty,
            implicit_begin=empty,
            implicit_end=empty,
            implicit_last=empty,
            implicit_counts=empty,
            frame_baseline=integers(frame_baseline),
            modified_frames=empty,
            frame_groups=integers(frame_groups),
            patch_last=integers(patch_last),
            patch_counts=integers(patch_counts),
            patch_roots=integers(patch_roots),
            patch_positions=integers(patch_positions),
            roots=integers(roots),
        )
