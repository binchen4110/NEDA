import ast

import argparse
import copy
import hashlib
import json
import os
import pickle
import sys
import time
import zlib
from collections import OrderedDict
from pathlib import Path
import numpy as np

SCHEMA = 1


def _rule_fingerprint(name):
    path = Path(__file__).resolve().parent
    content = (path / name).read_bytes()
    rules = json.loads((path / "cache_compatibility.json").read_text())

    class Canonical(ast.NodeTransformer):

        def strip(self, node):
            self.generic_visit(node)
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                node.body.pop(0)
            return node

        visit_Module = visit_FunctionDef = visit_AsyncFunctionDef = visit_ClassDef = strip

    tree = Canonical().visit(ast.parse(content))
    semantic = hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()
    saved = rules.get(name, {})
    if semantic == saved.get("semantic_sha256"):
        return saved["source_sha256"]
    return hashlib.sha256(content).hexdigest()


def signature(metadata, options):
    code = Path(__file__).resolve().parent
    keys = ("hops", "neighbors", "decay", "seed", "time_unit", "method_semantics")
    return {
        "schema": SCHEMA,
        "num_entities": metadata["num_entities"],
        "num_relations": metadata["num_relations"],
        "data": metadata["source_hashes"],
        "options": {k: options[k] for k in keys},
        "rules": {
            f: _rule_fingerprint(f) for f in ("data.py", "fast.py", "patch_gru.py", "preprocess.py")
        },
        "python": list(sys.version_info[:2]),
        "numpy": np.__version__,
    }


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def location(directory, sig):
    return Path(directory) / ".neda_preprocessed" / digest(sig)


def atomic_bytes(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_bytes(data)
    os.replace(temp, path)


def atomic_json(path, data):
    atomic_bytes(path, (json.dumps(data, ensure_ascii=False, indent=2) + "\n").encode())


def load_history(directory, snapshots, metadata, options):

    from .fast import FastHistory

    sig = signature(metadata, options)
    root = location(directory, sig)
    manifest = root / "manifest.json"
    if manifest.exists():
        info = json.loads(manifest.read_text())
        if info.get("signature") != sig:
            raise ValueError(
                "Preprocessing manifest does not match its content-addressed directory"
            )
        data = (root / "history.pkl.z").read_bytes()
        if hashlib.sha256(data).hexdigest() != info["history_sha256"]:
            raise ValueError("Preprocessed history checksum mismatch; rebuild this cache")
        history = pickle.loads(zlib.decompress(data))
        history.snapshots = snapshots
        history.plan_cache_bytes = options["plan_cache_bytes"]
        history.graph_cache = options["graph_cache"]
        history._plans = OrderedDict()
        history._plan_bytes = 0
        history.plan_hits = history.plan_builds = 0
        history._disk_cache_dir = str(root)
        history._disk_signature = digest(sig)
        history.disk_plan_hits = history.disk_plan_misses = 0
        history._relation_metadata_budget = min(history.plan_cache_bytes // 8, 256 * 1024**2)
        while (
            history._relation_metadata
            and history._relation_metadata_bytes > history._relation_metadata_budget
        ):
            _, old = history._relation_metadata.popitem(last=False)
            history._relation_metadata_bytes -= sum((x.nbytes for x in old))
        print(
            json.dumps({"event": "preprocessed_history_loaded", "directory": str(root)}), flush=True
        )
        return history
    history = FastHistory(snapshots, metadata["num_entities"], **options)
    history._disk_cache_dir = None
    history.disk_plan_hits = history.disk_plan_misses = 0
    return history


def plan_name(key):
    return f"{key[0]:06d}-{digest(key)[:24]}.pkl.z"


def read_plan(history, key):
    root = getattr(history, "_disk_cache_dir", None)
    if root is None:
        return None
    path = Path(root) / "plans" / plan_name(key)
    if not path.exists():
        history.disk_plan_misses += 1
        return None
    payload = pickle.loads(zlib.decompress(path.read_bytes()))
    if payload["signature"] != history._disk_signature or payload["key"] != key:
        raise ValueError("Execution-plan cache key mismatch")
    plan = payload["plan"]
    history.baseline_chunks(key[0], key[3], key[4])
    pool = history._baseline_pools[key[3], key[4]][0]
    plan.baseline = [pool[item] if kind == "pool" else item for kind, item in payload["baseline"]]
    history.disk_plan_hits += 1
    return plan


def write_plan(history, key, plan, root, sig):
    pool = history._baseline_pools[key[3], key[4]][0]
    indices = {id(c): i for i, c in enumerate(pool)}
    baseline = [
        ("pool", indices[id(c)]) if id(c) in indices else ("inline", c) for c in plan.baseline
    ]
    compact = copy.copy(plan)
    compact.baseline = []
    payload = {"signature": digest(sig), "key": key, "plan": compact, "baseline": baseline}
    data = zlib.compress(pickle.dumps(payload, protocol=5), level=1)
    path = root / "plans" / plan_name(key)
    atomic_bytes(path, data)
    return len(data)


def prepare(
    directory,
    *,
    seed=2024,
    hops=2,
    node_budget=8192,
    edge_budget=65536,
    cutoffs="all",
    output_dir=None,
):
    from .data import load_dataset
    from .train import NEIGHBORS

    started = time.monotonic()
    directory = Path(directory).resolve()
    snapshots, spans, metadata = load_dataset(directory)
    options = dict(
        hops=hops,
        neighbors=NEIGHBORS[directory.name],
        decay=1.0,
        seed=seed,
        graph_cache=65536,
        plan_cache_bytes=8 * 1024**3,
        time_unit=int(min(np.diff([s.time for s in snapshots[: spans["train"][1]]]))),
        method_semantics="paper-event-v1",
    )
    sig = signature(metadata, options)
    root = location(directory, sig)
    history = load_history(directory, snapshots, metadata, options)
    if not (root / "manifest.json").exists():
        history.baseline_chunks(len(snapshots) - 1, node_budget, edge_budget)
        for i in range(len(snapshots)):
            history.relation_frame(i, metadata["num_relations"])
        heads = np.arange(metadata["num_entities"])
        history.neighbor_mean_edges(len(snapshots) - 1, heads, 10000)
        history.neighbor_relation_time_edges(len(snapshots) - 1, heads, 10000)
        history._plans.clear()
        history._plan_bytes = 0
        data = zlib.compress(pickle.dumps(history, protocol=5), level=1)
        atomic_bytes(root / "history.pkl.z", data)
        atomic_json(
            root / "manifest.json",
            {
                "signature": sig,
                "history_sha256": hashlib.sha256(data).hexdigest(),
                "history_bytes": len(data),
                "kind": "data-only indices, graph chunks, occurrences, first-seen support, relation memberships",
                "window_policy": "stores full event indexes; runtime still applies strict cutoff and requested history window",
            },
        )
    selected = (
        list(range(1, len(snapshots)))
        if cutoffs == "all"
        else [] if cutoffs == "none" else sorted(set((int(x) for x in cutoffs.split(","))))
    )
    if any((not 0 <= i < len(snapshots) for i in selected)):
        raise ValueError("Cutoff outside dataset")
    written = existing = total_bytes = 0
    for j, cutoff in enumerate(selected, 1):
        roots = history.active_entities(cutoff)
        key = (cutoff, tuple(roots), False, node_budget, edge_budget)
        path = root / "plans" / plan_name(key)
        if path.exists():
            existing += 1
            total_bytes += path.stat().st_size
        else:
            plan = history.execution_plan(cutoff, roots, False, node_budget, edge_budget)
            total_bytes += write_plan(history, key, plan, root, sig)
            written += 1
        if j % 25 == 0 or j == len(selected):
            progress = {
                "dataset": directory.name,
                "cache_dir": str(root),
                "completed": j,
                "total": len(selected),
                "written": written,
                "existing": existing,
                "selected_plan_bytes": total_bytes,
                "seconds": time.monotonic() - started,
            }
            atomic_json(root / "progress.json", progress)
            print(json.dumps(progress), flush=True)
    result = {
        "dataset": directory.name,
        "cache_dir": str(root),
        "signature": sig,
        "cutoffs": selected,
        "written": written,
        "existing": existing,
        "selected_plan_bytes": total_bytes,
        "seconds": time.monotonic() - started,
        "history_bytes": (root / "history.pkl.z").stat().st_size,
    }
    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        atomic_json(out / "results.json", result)
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--hops", type=int, default=2)
    parser.add_argument("--node-budget", type=int, default=8192)
    parser.add_argument("--edge-budget", type=int, default=65536)
    parser.add_argument(
        "--cutoffs", default="all", help="all, none, or comma-separated prediction cutoffs"
    )
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    prepare(
        args.data_dir,
        seed=args.seed,
        hops=args.hops,
        node_budget=args.node_budget,
        edge_budget=args.edge_budget,
        cutoffs=args.cutoffs,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
