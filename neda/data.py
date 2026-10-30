from collections import defaultdict
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
import hashlib

import numpy as np


DEFAULT_HISTORY_LEN = 10000


@dataclass(frozen=True, order=True)
class ContactEvent:

    snapshot: int
    neighbor: int
    relation: int


@dataclass(frozen=True)
class Patch:
    explicit: bool
    start: int
    stop: int
    centers: tuple
    sampled_events: tuple = ()


@dataclass
class Graph:
    nodes: np.ndarray
    source: np.ndarray
    destination: np.ndarray
    relation: np.ndarray
    center_indices: np.ndarray


class Snapshot:
    def __init__(self, time, triples, num_relations):
        self.time = int(time)
        self.triples = np.asarray(triples, dtype=np.int64).reshape(-1, 3)
        self.adjacency = defaultdict(list)
        for s, r, o in self.triples:
            self.adjacency[int(s)].append((int(o), int(r)))
            self.adjacency[int(o)].append((int(s), int(r) + num_relations))
        self.entities = frozenset(self.adjacency)

    def graph(self):
        nodes = self.entities
        ids = np.array(sorted(nodes), dtype=np.int64)
        local = {int(n): i for i, n in enumerate(ids)}
        source, destination, relation = [], [], []
        for s in ids:
            for o, r in self.adjacency.get(int(s), ()):
                if o in local:
                    source.append(local[o])
                    destination.append(local[int(s)])
                    relation.append(r)
        return Graph(
            ids,
            np.array(source, dtype=np.int64),
            np.array(destination, dtype=np.int64),
            np.array(relation, dtype=np.int64),
            np.arange(len(ids), dtype=np.int64),
        )


class History:
    def __init__(
        self,
        snapshots,
        num_entities,
        *,
        hops=2,
        neighbors=4,
        decay=1.0,
        seed=0,
        graph_cache=2048,
        time_unit=None,
        method_semantics="paper-event-v1",
    ):
        if hops < 1 or neighbors < 1 or decay <= 0 or graph_cache < 0:
            raise ValueError("hops/neighbors/decay must be positive; cache must be nonnegative")
        self.snapshots = snapshots
        if method_semantics != "paper-event-v1":
            raise ValueError(f"Unknown method semantics: {method_semantics}")
        self.method_semantics = method_semantics
        self.num_entities = num_entities
        self.hops, self.neighbors, self.decay, self.seed = hops, neighbors, decay, seed
        self.graph_cache = graph_cache
        self.occurrences = [[] for _ in range(num_entities)]
        self.contacts = [defaultdict(list) for _ in range(num_entities)]
        self.events = [[] for _ in range(num_entities)]
        self.active_prefix = []
        active = set()
        for i, snap in enumerate(snapshots):
            if i and snapshots[i - 1].time >= snap.time:
                raise ValueError("Snapshots must have distinct, increasing timestamps")
            active.update(snap.entities)
            self.active_prefix.append(tuple(sorted(active)))
            for e in snap.entities:
                self.occurrences[e].append(i)
                for o, r in sorted(snap.adjacency[e]):
                    self.contacts[e][o].append(i)
                    self.events[e].append(ContactEvent(i, o, r))
        self._event_snapshots = [
            np.array([e.snapshot for e in events], dtype=np.int64) for events in self.events
        ]

        neighbor_pairs = [
            (entity, other, times[0])
            for entity, contacts in enumerate(self.contacts)
            for other, times in sorted(contacts.items())
        ]
        pairs = np.asarray(neighbor_pairs, dtype=np.int64).reshape(-1, 3)
        self._neighbor_receivers, self._neighbor_sources, self._neighbor_first = pairs.T
        self.time_unit = time_unit or (
            snapshots[1].time - snapshots[0].time if len(snapshots) > 1 else 1
        )
        if self.time_unit <= 0:
            raise ValueError("time_unit must be positive")

    def active_entities(self, cutoff):
        return self.active_prefix[cutoff - 1] if cutoff else ()

    def neighbor_history_window(self, cutoff, history_len):

        if not 0 <= cutoff <= len(self.snapshots):
            raise ValueError("Invalid history cutoff")
        if history_len is not None and history_len < 1:
            raise ValueError("history_len must be positive")
        past = range(cutoff)
        return past[-history_len:] if history_len is not None else past

    def neighbor_mean_edges(self, cutoff, heads, history_len=None):

        window = self.neighbor_history_window(cutoff, history_len)
        mask = np.zeros(self.num_entities, dtype=bool)
        mask[np.asarray(heads, dtype=np.int64)] = True
        start = window.start
        if start == 0:
            keep = (self._neighbor_first < cutoff) & mask[self._neighbor_receivers]
            rows, columns = self._neighbor_receivers[keep], self._neighbor_sources[keep]
        else:

            pairs = []
            for entity in np.flatnonzero(mask):
                for other, times in sorted(self.contacts[entity].items()):
                    end = bisect_left(times, cutoff)
                    if end and times[end - 1] >= start:
                        pairs.append((entity, other))
            rows, columns = np.asarray(pairs, dtype=np.int64).reshape(-1, 2).T
        degree = np.bincount(rows, minlength=self.num_entities)
        weights = 1.0 / degree[rows]
        return rows, columns, weights


def directed_queries(snapshot, num_relations):
    original = snapshot.triples
    inverse = original[:, [2, 1, 0]].copy()
    inverse[:, 1] += num_relations
    return np.concatenate([original, inverse])


def load_dataset(directory):
    directory = Path(directory)
    stats = [int(n) for n in (directory / "stat.txt").read_text().split()]
    num_entities, num_relations = stats[:2]
    arrays, source_hashes = {}, {}
    for split in ("train", "valid", "test"):
        path = directory / f"{split}.txt"
        source_hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
        a = np.loadtxt(path, dtype=np.int64, usecols=(0, 1, 2, 3), ndmin=2)
        if len(a) == 0 or a.min() < 0:
            raise ValueError(f"{split}: empty split or negative IDs")
        if a[:, [0, 2]].max() >= num_entities or a[:, 1].max() >= num_relations:
            raise ValueError(f"{split}: ID outside stat.txt range")
        arrays[split] = a
    if not (
        arrays["train"][:, 3].max() < arrays["valid"][:, 3].min()
        and arrays["valid"][:, 3].max() < arrays["test"][:, 3].min()
    ):
        raise ValueError("Temporal split overlap: refusing an ambiguous evaluation protocol")
    source_stats = {
        s: {
            "facts": len(a),
            "snapshots": len(np.unique(a[:, 3])),
            "first": int(a[:, 3].min()),
            "last": int(a[:, 3].max()),
        }
        for s, a in arrays.items()
    }
    snapshots, ranges = [], {}
    for split, a in arrays.items():
        begin = len(snapshots)
        for time in np.unique(a[:, 3]):
            snapshots.append(Snapshot(time, a[a[:, 3] == time, :3], num_relations))
        ranges[split] = (begin, len(snapshots))
    observed_entities = set(
        np.concatenate([a[:, [0, 2]].ravel() for a in arrays.values()]).tolist()
    )
    metadata = {
        "num_entities": num_entities,
        "num_relations": num_relations,
        "source_stat_entities": stats[0],
        "source": str(directory.resolve()),
        "source_hashes": source_hashes,
        "source_splits": source_stats,
        "smoke_subset": False,
        "retained_original_entity_ids": None,
        "observed_entities": len(observed_entities),
        "splits": {
            s: {"facts": len(a), "snapshots": ranges[s][1] - ranges[s][0]}
            for s, a in arrays.items()
        },
    }
    return snapshots, ranges, metadata
