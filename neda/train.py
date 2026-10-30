import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.nn import functional as F
from .data import DEFAULT_HISTORY_LEN, directed_queries, load_dataset
from .model import NEDA, MODEL_VERSION, NEIGHBOR_DROPOUT, FUSION_DROPOUT
from .static_graph import load_static_graph

NEIGHBORS = {"ICEWS14s": 4, "ICEWS18": 4, "ICEWS05-15": 11, "GDELT": 8, "ICEWS22": 4, "ICEWS23": 4}


def resolve_history_len(requested, previous=None):
    if requested is not None and requested < 1:
        raise ValueError("history_len must be positive")
    if previous is None:
        return DEFAULT_HISTORY_LEN if requested is None else requested
    names = [
        key for key in ("history_len", "neighbor_history_len", "identity_len") if key in previous
    ]
    saved = previous[names[0]] if names else None
    if any((previous[key] != saved for key in names)):
        raise ValueError("Conflicting history window fields in checkpoint")
    if saved is not None and saved < 1:
        raise ValueError("Checkpoint history_len must be positive")
    if requested is not None and requested != saved:
        raise ValueError("history_len differs from checkpoint; start a new run")
    return saved


def resolve_query_batch_size(requested, previous=None):
    if requested is not None and requested < 0:
        raise ValueError("query_batch_size must be nonnegative (0 means full snapshot)")
    saved = previous.get("query_batch_size", 1024) if previous is not None else 0
    if previous is not None and requested is not None and (requested != saved):
        raise ValueError("Query batching differs from checkpoint; start a new run")
    return saved if requested is None else requested


def query_batches(queries, batch_size):
    if batch_size < 0:
        raise ValueError("Query batch size must be nonnegative")
    if not len(queries):
        return
    step = batch_size or len(queries)
    for begin in range(0, len(queries), step):
        yield queries[begin : begin + step]


def configure_fixed_architecture(args, previous=None):
    fixed = {
        "head_neighbor": "last-fusion-shared-gru",
        "neighbor_attention": "query-rel-time",
        "prediction_directions": "joint",
        "method_semantics": "paper-event-v1",
        "decoder_semantics": "tirgn",
        "neighbor_dropout": NEIGHBOR_DROPOUT,
        "fusion_dropout": FUSION_DROPOUT,
        "decay": 1.0,
    }
    if previous is not None:
        incompatible = {
            key: previous.get(key) for key, value in fixed.items() if previous.get(key) != value
        }
        if incompatible:
            raise ValueError(
                f"Checkpoint architecture incompatible with the current model: {incompatible}. Use the source snapshot saved with that historical run, or start a new run."
            )
    for key, value in fixed.items():
        supplied = getattr(args, key, value)
        if supplied is not None and supplied != value:
            raise ValueError(f"{key} is fixed to {value}; architecture switches were removed")
        setattr(args, key, value)


def ranks(scores, queries, answers):
    target = scores.gather(1, queries[:, 2:3])
    raw = 1 + (scores > target).sum(1) + 0.5 * ((scores == target).sum(1) - 1)
    filtered = scores.clone()
    rows, columns = ([], [])
    for i, (s, r, o) in enumerate(queries.detach().cpu().tolist()):
        other = answers[s, r] - {o}
        rows.extend([i] * len(other))
        columns.extend(other)
    if rows:
        filtered[rows, columns] = -torch.inf
    filt = 1 + (filtered > target).sum(1) + 0.5 * ((filtered == target).sum(1) - 1)
    return (raw, filt)


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        raise ValueError("Cannot report an empty evaluation")
    return {
        "MRR": float(np.mean(1.0 / values)),
        **{f"Hits@{k}": float(np.mean(values <= k)) for k in (1, 3, 10)},
        "queries": len(values),
    }


@torch.no_grad()
def evaluate(model, history, span, num_relations, query_batch_size=0):
    model.eval()
    raw, filtered = [], []
    vocabulary = history.vocabulary(span[0])
    for cutoff in range(*span):
        snapshot = history.snapshots[cutoff]
        entities, relations = model.encode_for_prediction(history, cutoff)
        queries = directed_queries(snapshot, num_relations)
        neighbors = model.prepare_neighbors(
            entities, history, cutoff, np.unique(queries[:, 0]), relations=relations
        )
        answers = defaultdict(set)
        for s, r, o in queries:
            answers[int(s), int(r)].add(int(o))
        for batch in query_batches(queries, query_batch_size):
            q = torch.as_tensor(batch, device=entities.device)
            timestamp = snapshot.time // history.time_unit
            logp = model.log_probabilities(entities, q, timestamp, vocabulary, neighbors)
            a, b = ranks(logp, q, answers)
            raw.extend(a.cpu().tolist())
            filtered.extend(b.cpu().tolist())
        for s, r, o in queries:
            vocabulary[int(s), int(r)].add(int(o))
    return {"raw": summarize(raw), "time_filtered": summarize(filtered)}


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def run(args):
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "config.json").exists() and (not args.resume):
        raise ValueError(f"Refusing to overwrite an existing run: {output}; use a fresh directory")
    state = torch.load(args.resume, map_location="cpu", weights_only=False) if args.resume else None
    if state and state["config"].get("fusion_semantics", {}).get("version") != MODEL_VERSION:
        raise ValueError(
            "Checkpoint fusion differs from independent-static relation-GRU fusion. Use the checkpoint source snapshot or start a new run."
        )
    configure_fixed_architecture(args, state["config"] if state else None)
    args.lr_scheduler = "none"
    if state and state["config"].get("lr_scheduler", "none") != "none":
        raise ValueError("Checkpoint scheduler differs from this NEDA version")
    previous = state["config"] if state else {}
    for name in ("ema_decay", "label_smoothing"):
        if previous.get(name, 0.0):
            raise ValueError(f"Checkpoint {name} differs from this NEDA version")
        setattr(args, name, 0.0)
    saved_rate = previous.get("history_rate", 0.3)
    if state and args.history_rate is not None and (args.history_rate != saved_rate):
        raise ValueError("history_rate differs from checkpoint")
    args.history_rate = saved_rate if args.history_rate is None else args.history_rate
    if not 0 < args.history_rate < 1:
        raise ValueError("history_rate must be in (0,1)")
    args.history_len = resolve_history_len(DEFAULT_HISTORY_LEN, state["config"] if state else None)
    args.query_batch_size = resolve_query_batch_size(
        args.query_batch_size, state["config"] if state else None
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    if args.device.startswith("cuda") and (not torch.cuda.is_available()):
        raise ValueError("CUDA requested but unavailable; choose --device cpu explicitly")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
    data_dir = (
        Path(args.data_dir).resolve()
        if args.data_dir
        else Path(__file__).resolve().parents[1] / "data" / args.dataset
    )
    args.smoke = False
    snapshots, spans, metadata = load_dataset(data_dir)
    static_graph = load_static_graph(data_dir, metadata)
    static_metadata = dict(static_graph.metadata, num_blocks=math.gcd(args.dimension, 100))
    if state and state["config"].get("static_initialization") != static_metadata:
        raise ValueError(
            "Static initialization or its data differs from checkpoint; start a new run"
        )
    train_times = [s.time for s in snapshots[: spans["train"][1]]]
    time_unit = min(np.diff(train_times)) if len(train_times) > 1 else 1
    history_options = {"plan_cache_bytes": int(args.plan_cache_gib * 1024**3)}
    history_kwargs = dict(
        hops=args.hops,
        neighbors=NEIGHBORS[args.dataset],
        decay=args.decay,
        seed=args.seed,
        graph_cache=args.graph_cache,
        time_unit=int(time_unit),
        method_semantics=args.method_semantics,
        **history_options,
    )
    from .preprocess import load_history

    history = load_history(data_dir, snapshots, metadata, history_kwargs)
    args.no_implicit = True
    args.no_positions = args.no_inter_attention = args.no_temporal = False
    config = vars(args).copy()
    config["query_batch_policy"] = (
        "full_snapshot" if args.query_batch_size == 0 else "fixed_max_size"
    )
    config["resolved_neighbors"] = history.neighbors
    config["time_unit_from_training"] = history.time_unit
    config["data"] = metadata
    config["static_initialization"] = static_metadata
    config["fusion_semantics"] = {"version": MODEL_VERSION}
    config["evaluation_protocol"] = (
        "rolling ground truth; both directions; time-filtered; average ties"
    )
    model = NEDA(
        metadata["num_entities"],
        metadata["num_relations"],
        len(snapshots) + 1,
        dimension=args.dimension,
        layers=args.layers,
        history_rate=args.history_rate,
        dropout=args.dropout,
        channels=args.channels,
        activation_checkpoint=not args.no_activation_checkpoint,
        encoder_activation_cache_gib=args.encoder_activation_cache_gib,
        graph_node_budget=args.graph_node_budget,
        graph_edge_budget=args.graph_edge_budget,
        static_graph=static_graph,
    ).to(args.device)
    print(
        json.dumps(
            {
                "event": "ready",
                "dataset": args.dataset,
                "smoke": args.smoke,
                "splits": metadata["splits"],
                "entities": metadata["num_entities"],
                "parameters": sum((p.numel() for p in model.parameters())),
                "device": args.device,
                "method_semantics": args.method_semantics,
                "decoder_semantics": args.decoder_semantics,
                "head_neighbor": args.head_neighbor,
                "prediction_directions": args.prediction_directions,
                "neighbor_attention": args.neighbor_attention,
                "static_initialization": static_metadata,
                "neighbor_dropout": args.neighbor_dropout,
                "fusion_dropout": args.fusion_dropout,
                "lr_scheduler": args.lr_scheduler,
                "history_len": args.history_len,
                "query_batch_size": args.query_batch_size,
                "query_batch_policy": config["query_batch_policy"],
                "fusion_version": MODEL_VERSION,
                "output_dir": str(output),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_epoch, best, stale = (0, -1.0, 0)
    if args.resume:
        previous = state["config"]
        keys = [
            "dataset",
            "dimension",
            "layers",
            "hops",
            "history_rate",
            "dropout",
            "channels",
            "seed",
            "decay",
            "smoke",
            "no_implicit",
            "no_positions",
            "no_inter_attention",
            "no_temporal",
        ]
        previous_data = {k: v for k, v in previous["data"].items() if k != "source"}
        current_data = {k: v for k, v in metadata.items() if k != "source"}
        if (
            any((previous[k] != config[k] for k in keys))
            or previous_data != current_data
            or previous["resolved_neighbors"] != history.neighbors
        ):
            raise ValueError("Resume configuration or dataset differs from checkpoint")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch, best = (state["epoch"] + 1, state["best_validation_MRR"])
        stale = state.get("stale", 0)
        random.setstate(state["random_state"])
        np.random.set_state(state["numpy_state"])
        torch.set_rng_state(state["torch_state"].cpu())
        if torch.cuda.is_available() and state["cuda_state"] is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_state"]])
            old_device = torch.device(previous["device"])
            if old_device.type == "cuda" and args.device.startswith("cuda"):
                torch.cuda.set_rng_state(
                    state["cuda_state"][old_device.index or 0].cpu(), device=args.device
                )
        if not args.test_only and (not (output / "best.pt").exists()):
            previous_best = Path(args.resume).resolve().parent / "best.pt"
            if not previous_best.exists():
                raise ValueError("Training resume requires the best.pt companion checkpoint")
            best_state = torch.load(previous_best, map_location="cpu", weights_only=False)
            torch.save(best_state, output / "best.pt")
    save_json(output / "config.json", config)
    if not args.test_only:
        for epoch in range(start_epoch, args.epochs):
            epoch_lrs = [group["lr"] for group in optimizer.param_groups]
            model.train()
            order = list(range(1, spans["train"][1]))
            random.shuffle(order)
            loss_sum, nll_sum, n_queries = (0.0, 0.0, 0)
            for step, cutoff in enumerate(order, 1):
                started = time.monotonic()
                print(
                    json.dumps(
                        {
                            "event": "train_step_start",
                            "epoch": epoch + 1,
                            "step": step,
                            "total": len(order),
                            "history_snapshots": cutoff,
                        }
                    ),
                    flush=True,
                )
                optimizer.zero_grad(set_to_none=True)
                entities, relations = model.encode_for_prediction(history, cutoff)
                if args.phase_timing and args.device.startswith("cuda"):
                    torch.cuda.synchronize(args.device)
                encoded_at = time.monotonic()
                queries = directed_queries(snapshots[cutoff], metadata["num_relations"])
                query_count = len(queries)
                vocabulary = history.query_vocabulary(cutoff)
                total_loss = entities.sum() * 0.0
                neighbors = model.prepare_neighbors(
                    entities, history, cutoff, np.unique(queries[:, 0]), relations=relations
                )
                for batch in query_batches(queries, args.query_batch_size):
                    q = torch.as_tensor(batch, device=args.device)
                    timestamp = snapshots[cutoff].time // history.time_unit
                    logp = model.log_probabilities(entities, q, timestamp, vocabulary, neighbors)
                    # 论文 §3.4 Learning Objective：交叉熵／负对数似然（式 5）。
                    total_loss = (
                        total_loss + F.nll_loss(logp, q[:, 2], reduction="sum") / query_count
                    )
                if not torch.isfinite(total_loss):
                    raise FloatingPointError("Nonfinite loss; refusing to continue")
                if args.phase_timing and args.device.startswith("cuda"):
                    torch.cuda.synchronize(args.device)
                decoded_at = time.monotonic()
                total_loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_norm, error_if_nonfinite=True
                )
                optimizer.step()
                if args.phase_timing and args.device.startswith("cuda"):
                    torch.cuda.synchronize(args.device)
                loss_sum += total_loss.item() * query_count
                nll_sum += total_loss.item() * query_count
                n_queries += query_count
                print(
                    json.dumps(
                        {
                            "event": "train_step",
                            "epoch": epoch + 1,
                            "step": step,
                            "loss": total_loss.item(),
                            "gradient_norm": float(norm),
                            "seconds": time.monotonic() - started,
                            **(
                                {
                                    "encode_seconds": encoded_at - started,
                                    "decode_seconds": decoded_at - encoded_at,
                                    "backward_optimizer_seconds": time.monotonic() - decoded_at,
                                    "plan_cache_mib": getattr(history, "_plan_bytes", 0) / 1024**2,
                                }
                                if args.phase_timing
                                else {}
                            ),
                        }
                    ),
                    flush=True,
                )
            validation = evaluate(
                model,
                history,
                spans["valid"],
                metadata["num_relations"],
                args.query_batch_size,
            )
            score = validation["time_filtered"]["MRR"]
            improved = score > best
            stale = 0 if improved else stale + 1
            best = max(best, score)
            next_lrs = [group["lr"] for group in optimizer.param_groups]
            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "ema_state": None,
                "lr_scheduler_state": None,
                "epoch": epoch,
                "best_validation_MRR": best,
                "config": config,
                "stale": stale,
                "random_state": random.getstate(),
                "numpy_state": np.random.get_state(),
                "torch_state": torch.get_rng_state(),
                "cuda_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            }
            torch.save(state, output / "last.pt")
            if improved:
                torch.save(state, output / "best.pt")
            record = {
                "epoch": epoch + 1,
                "train_nll": nll_sum / max(1, n_queries),
                "train_loss": loss_sum / max(1, n_queries),
                "validation": validation,
                "evaluation_model": "online",
                "lr": epoch_lrs,
                "next_lr": next_lrs,
            }
            record["plan_cache"] = {
                "builds": history.plan_builds,
                "hits": history.plan_hits,
                "mib": history._plan_bytes / 1024**2,
            }
            if args.device.startswith("cuda"):
                record["cuda_memory_gib"] = {
                    "peak_allocated": torch.cuda.max_memory_allocated(args.device) / 1024**3,
                    "peak_reserved": torch.cuda.max_memory_reserved(args.device) / 1024**3,
                }
            with (output / "metrics.jsonl").open("a") as file:
                file.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
            if (epoch + 1) % 3 == 0:
                recent = [
                    json.loads(line)
                    for line in (output / "metrics.jsonl").read_text().splitlines()[-3:]
                ]
                report = {
                    "event": "three_epoch_report",
                    "prediction_directions": args.prediction_directions,
                    "epochs": recent,
                }
                with (output / "validation_reports.jsonl").open("a") as file:
                    file.write(json.dumps(report) + "\n")
                print(json.dumps(report), flush=True)
            if args.patience and stale >= args.patience:
                break
    best_path = Path(args.resume) if args.test_only else output / "best.pt"
    if not best_path.exists():
        raise ValueError("No checkpoint available; run training first")
    state = torch.load(best_path, map_location=args.device, weights_only=False)
    model.load_state_dict(state["model"])
    result = {
        "dataset": args.dataset,
        "smoke_subset": args.smoke,
        "static_initialization": static_metadata,
        "method_semantics": args.method_semantics,
        "decoder_semantics": args.decoder_semantics,
        "head_neighbor": args.head_neighbor,
        "prediction_directions": args.prediction_directions,
        "neighbor_attention": args.neighbor_attention,
        "neighbor_dropout": args.neighbor_dropout,
        "fusion_dropout": args.fusion_dropout,
        "lr_scheduler": args.lr_scheduler,
        "ema_decay": args.ema_decay,
        "label_smoothing": args.label_smoothing,
        "history_rate": args.history_rate,
        "evaluation_model": "online",
        "history_len": args.history_len,
        "query_batch_size": args.query_batch_size,
        "query_batch_policy": config["query_batch_policy"],
        "fusion_semantics": config["fusion_semantics"],
        "best_epoch": state["epoch"] + 1,
        "best_validation_MRR": state["best_validation_MRR"],
        "test": evaluate(
            model,
            history,
            spans["test"],
            metadata["num_relations"],
            args.query_batch_size,
        ),
    }
    save_json(output / "results.json", result)
    print(json.dumps(result), flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=NEIGHBORS, default="ICEWS14s")
    p.add_argument("--data-dir")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--dimension", type=int, default=200)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--hops", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--weight-decay", type=float, default=1e-05)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--channels", type=int, default=50)
    p.add_argument("--history-rate", type=float)
    p.add_argument("--grad-norm", type=float, default=1.0)
    p.add_argument("--query-batch-size", type=int)
    p.add_argument("--graph-cache", type=int, default=65536)
    p.add_argument("--plan-cache-gib", type=float, default=8.0)
    p.add_argument("--phase-timing", action="store_true")
    p.add_argument("--graph-node-budget", type=int, default=8192)
    p.add_argument("--graph-edge-budget", type=int, default=65536)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--no-activation-checkpoint", action="store_true")
    p.add_argument("--encoder-activation-cache-gib", type=float, default=0.0)
    p.add_argument("--resume")
    p.add_argument("--test-only", action="store_true")
    return p


def main():
    arguments = parser().parse_args()
    if arguments.epochs < 1:
        raise ValueError("Epochs must be positive")
    if arguments.test_only and (not arguments.resume):
        raise ValueError("--test-only requires --resume")
    run(arguments)


if __name__ == "__main__":
    main()
