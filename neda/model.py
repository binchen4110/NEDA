import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .neighbors import Neighbors, NeighborAttention
from .static_graph import StaticInit
from .data import DEFAULT_HISTORY_LEN
from .initialization import preserve_initialization_rng

MODEL_VERSION = "neda-last6-gru-last-patch-residual-no-w1-v1"
NEIGHBOR_DROPOUT = 0.0
FUSION_DROPOUT = 0.1


class NeighborFusion(nn.Module):

    def __init__(self, dimension):
        super().__init__()
        self.h_proj = nn.Linear(dimension, dimension, bias=False)
        self.n_proj = nn.Linear(dimension, dimension, bias=False)
        preserve_initialization_rng(dimension, 1)
        self.proj = nn.Linear(dimension, dimension, bias=False)
        self.gru = nn.GRUCell(dimension, dimension)
        self.neighbor_drop = nn.Dropout(NEIGHBOR_DROPOUT)
        self.dropout = nn.Dropout(FUSION_DROPOUT)

    def forward(self, h, neighbor):
        neighbor = self.neighbor_drop(neighbor)
        mixed = self.h_proj(h) + self.n_proj(neighbor)
        x = F.relu(self.proj(mixed))
        x = F.normalize(self.dropout(x), p=2, dim=-1)
        hidden = F.normalize(h, p=2, dim=-1)
        return F.normalize(self.gru(x, hidden), p=2, dim=-1)

    def forward_grouped(self, h, neighbor, sizes):
        if self.training and self.neighbor_drop.p:
            return torch.cat([self(a, b) for a, b in zip(h.split(sizes), neighbor.split(sizes))])
        mixed = self.h_proj(h) + self.n_proj(self.neighbor_drop(neighbor))
        x = F.relu(self.proj(mixed))
        x = torch.cat([self.dropout(x) for x in x.split(sizes)])
        x = F.normalize(x, p=2, dim=-1)
        return F.normalize(self.gru(x, F.normalize(h, p=2, dim=-1)), p=2, dim=-1)


# 论文 §3.2 Intra-Patch Encoder — Structural Dependency（式 1）。
class RelationAttention(nn.Module):

    def __init__(self, dimension):
        super().__init__()
        self.W1 = nn.Linear(3 * dimension, dimension, bias=False)
        self.a = nn.Linear(dimension, 1, bias=False)
        self.W2 = nn.Linear(dimension, dimension, bias=False)

    def forward(self, h, relations, source, destination, edge_relation):
        result = self.W2(h)
        if not len(source):
            return result
        r = relations.index_select(0, edge_relation)
        ws, wo, wr = self.W1.weight.split(h.shape[-1], dim=1)
        projected = (
            F.linear(h, ws).index_select(0, destination)
            + F.linear(h, wo).index_select(0, source)
            + F.linear(relations, wr).index_select(0, edge_relation)
        )
        logits = self.a(F.leaky_relu(projected, negative_slope=0.2)).squeeze(-1)
        maxima = h.new_full((len(h),), -torch.inf)
        maxima.scatter_reduce_(0, destination, logits.detach(), reduce="amax", include_self=True)
        weights = (logits - maxima[destination]).exp()
        denominators = h.new_zeros(len(h)).index_add(0, destination, weights)
        alpha = weights / denominators.index_select(0, destination)
        return result.index_add(0, destination, alpha[:, None] * (h.index_select(0, source) + r))


# 论文 §3.3 Inter-Patch Attention（式 3）。
class InterPatchAttention(nn.Module):

    def __init__(self, dimension, max_patches):
        super().__init__()
        self.position = nn.Embedding(max_patches, dimension)
        nn.init.normal_(self.position.weight, std=0.02)
        self.Q = nn.Parameter(torch.empty(1, dimension))
        nn.init.xavier_uniform_(self.Q)
        self.K = nn.Linear(dimension, dimension, bias=False)
        self.V = nn.Linear(dimension, dimension, bias=False)

    def forward(self, patches, groups, positions, num_entities):
        x = patches + self.position(positions)
        logits = (self.K(x) * self.Q).sum(-1) / math.sqrt(x.shape[-1])
        maxima = x.new_full((num_entities,), -torch.inf)
        maxima.scatter_reduce_(0, groups, logits.detach(), reduce="amax", include_self=True)
        weights = (logits - maxima[groups]).exp()
        denominator = x.new_zeros(num_entities).index_add(0, groups, weights)
        messages = self.V(x) * (weights / denominator[groups])[:, None]
        return x.new_zeros((num_entities, x.shape[-1])).index_add(0, groups, messages)


# 论文 §3.4 Event Prediction：时间卷积评分。
class TimeConvDecoder(nn.Module):

    def __init__(self, dimension, channels=50, dropout=0.2):
        super().__init__()
        self.bn0 = nn.BatchNorm1d(4)
        self.conv = nn.Conv1d(4, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.fc = nn.Linear(channels * dimension, dimension)
        self.bn2 = nn.BatchNorm1d(dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, entities, relations, time, queries, query_head_embeddings):
        candidates = entities.tanh()
        heads = query_head_embeddings.tanh()
        x = torch.stack([heads, relations[queries[:, 1]], *time], dim=1)
        x = self.dropout(self.bn0(x))
        x = self.dropout(F.relu(self.bn1(self.conv(x))))
        x = self.dropout(self.fc(x.flatten(1)))

        if len(queries) > 1:
            x = self.bn2(x)
        return F.relu(x) @ candidates.T


class DualDecoder(nn.Module):
    def __init__(self, dimension, *, history_rate=0.3, dropout=0.2, channels=50):
        super().__init__()
        if not 0 < history_rate < 1:
            raise ValueError("history_rate must be in (0,1)")
        self.history_rate = history_rate
        # 论文 §3.4 Dual Decoder：生成与重复两个预测分支。
        self.generate = TimeConvDecoder(dimension, channels, dropout)
        self.repeat = TimeConvDecoder(dimension, channels, dropout)
        self.linear_weight = nn.Parameter(torch.randn(1, dimension))
        self.linear_bias = nn.Parameter(torch.randn(1, dimension))
        self.period_weight = nn.Parameter(torch.randn(1, dimension))
        self.period_bias = nn.Parameter(torch.randn(1, dimension))

        with torch.random.fork_rng(devices=[]):
            self.history_gate = nn.Linear(2 * dimension + 1, 1)
            nn.init.zeros_(self.history_gate.weight)
            nn.init.constant_(self.history_gate.bias, math.log(history_rate / (1 - history_rate)))

    def forward(
        self,
        entities,
        relations,
        queries,
        time_index,
        history_mask,
        query_head_embeddings,
    ):
        t = entities.new_full((len(queries), 1), float(time_index))
        time = (
            t * self.linear_weight + self.linear_bias,
            torch.sin(t * self.period_weight + self.period_bias),
        )
        raw = F.log_softmax(
            self.generate(entities, relations, time, queries, query_head_embeddings),
            dim=-1,
        )
        logits = self.repeat(entities, relations, time, queries, query_head_embeddings)

        has_history = history_mask.any(dim=-1, keepdim=True)

        allowed = history_mask | ~has_history
        historical = F.log_softmax(logits.masked_fill(~allowed, -torch.inf), dim=-1)
        head = query_head_embeddings
        count = torch.log1p(history_mask.sum(-1, keepdim=True).to(entities.dtype))
        features = torch.cat((head.tanh(), relations[queries[:, 1]], count), dim=-1)
        logit = self.history_gate(features)

        mixed = torch.logaddexp(raw + F.logsigmoid(-logit), historical + F.logsigmoid(logit))
        return torch.where(has_history, mixed, raw)


class NEDA(nn.Module):
    def __init__(
        self,
        num_entities,
        num_relations,
        max_patches,
        *,
        dimension=200,
        layers=2,
        history_rate=0.3,
        dropout=0.2,
        channels=50,
        activation_checkpoint=True,
        graph_node_budget=8192,
        graph_edge_budget=65536,
        encoder_activation_cache_gib=0.0,
        static_graph=None,
    ):
        super().__init__()
        if not math.isfinite(encoder_activation_cache_gib) or encoder_activation_cache_gib < 0:
            raise ValueError("Activation cache budget must be finite and nonnegative")
        self.history_len = DEFAULT_HISTORY_LEN
        self.entity = nn.Embedding(num_entities, dimension)
        self.relation = nn.Embedding(2 * num_relations, dimension)
        nn.init.normal_(self.entity.weight)
        nn.init.xavier_normal_(self.relation.weight)
        self.structural = nn.ModuleList([RelationAttention(dimension) for _ in range(layers)])
        self.add_module("temporal", None)
        preserve_initialization_rng(dimension, 2)
        self.inter = InterPatchAttention(dimension, max_patches)
        self.decoder = DualDecoder(
            dimension,
            history_rate=history_rate,
            dropout=dropout,
            channels=channels,
        )
        self.activation_checkpoint = activation_checkpoint
        self.encoder_activation_cache_bytes = int(encoder_activation_cache_gib * 1024**3)
        self.graph_node_budget, self.graph_edge_budget = graph_node_budget, graph_edge_budget
        self.fusion = NeighborFusion(dimension)
        self.neighbor_attn = NeighborAttention(dimension)
        self.static = StaticInit(static_graph, dimension) if static_graph is not None else None
        from .relation_gru import RelationGRU

        self.rel_gru = RelationGRU(dimension)
        from .patch_gru import RecentPatchGRU

        with torch.random.fork_rng(devices=[]):
            self.temporal = RecentPatchGRU(dimension)

    def initial_entity_embeddings(self):

        if self.static is None:
            return self.entity.weight
        return self.static(self.entity.weight)

    # 论文 §3.1 Subgraph Extraction：共享快照计算各实体的局部结构表示。
    def _structural_chunks(self, chunks, *, initial_entities):
        device = initial_entities.device
        result = []
        for chunk in chunks:

            nodes, source, destination, rel = (
                torch.as_tensor(getattr(chunk, name), device=device)
                for name in (
                    "nodes",
                    "source",
                    "destination",
                    "relation",
                )
            )

            def evaluate(
                _weights,
                nodes=nodes,
                source=source,
                destination=destination,
                rel=rel,
            ):
                h = _weights.index_select(0, nodes)
                for layer in self.structural:
                    h = layer(h, self.relation.weight, source, destination, rel)
                return h

            keep_activations = (
                device.type == "cuda"
                and self.encoder_activation_cache_bytes > 0
                and torch.cuda.memory_allocated(device) < self.encoder_activation_cache_bytes
            )
            if self.training and self.activation_checkpoint and not keep_activations:
                result.append(checkpoint(evaluate, initial_entities, use_reentrant=False))
            else:
                result.append(evaluate(initial_entities))
        return (
            torch.cat(result)
            if result
            else self.entity.weight.new_empty((0, self.entity.embedding_dim))
        )

    def encode(self, history, cutoff, *, _cache=None):
        if not hasattr(history, "execution_plan") or history.hops < len(self.structural):
            raise ValueError("NEDA requires FastHistory with hops >= layers")
        roots = history.active_entities(cutoff)
        initial_entities = self.initial_entity_embeddings()
        if not roots:
            return initial_entities
        plan = history.execution_plan(
            cutoff, roots, False, self.graph_node_budget, self.graph_edge_budget
        )
        device = self.entity.weight.device

        def indices(name):
            return torch.as_tensor(getattr(plan, name), device=device)

        frames = self._structural_chunks(plan.baseline, initial_entities=initial_entities)
        if _cache is not None:

            _cache.update(history=history, cutoff=cutoff, frames=frames)

        # 论文 §3.2 Intra-Patch Encoder：patch 内时间聚合。
        structural = frames.index_select(0, indices("frame_baseline"))
        patch_initial = initial_entities[indices("roots")[indices("patch_roots")]]
        if plan.recent_gru is None or plan.recent_gru.limit != self.temporal.recent_patch_limit:
            raise ValueError("Invalid recent-patch GRU plan; rebuild preprocessing")
        patches = self.temporal(
            structural,
            indices("frame_groups"),
            indices("patch_counts"),
            initial_states=patch_initial,
            plan=plan.recent_gru,
        )
        final = self.inter(patches, indices("patch_roots"), indices("patch_positions"), len(roots))
        last = np.bincount(plan.patch_roots, minlength=len(roots)).cumsum() - 1
        final = final + patches[torch.as_tensor(last, device=device)]
        result = initial_entities.index_copy(0, indices("roots"), final)
        return result

    def encode_relations(self, history, cutoff, *, _cache=None):
        from .fast import pack_graphs

        window = list(history.neighbor_history_window(cutoff, self.history_len))
        if not window:
            return self.relation.weight
        graphs = [history._snapshot_graphs[i + 1] for i in window]
        if (
            _cache is not None
            and _cache.get("history") is history
            and _cache.get("cutoff") == cutoff
            and "frames" in _cache
        ):
            start, stop = history._snapshot_offsets[[window[0], window[-1] + 1]]
            structural = _cache["frames"][int(start) : int(stop)]
        else:
            chunks = pack_graphs(
                ((g, (), ()) for g in graphs), self.graph_node_budget, self.graph_edge_budget
            )
            initial = self.initial_entity_embeddings()
            structural = self._structural_chunks(
                chunks,
                initial_entities=initial,
            )

        structural = F.normalize(structural, p=2, dim=-1)
        metadata = [history.relation_frame(i, self.relation.num_embeddings // 2) for i in window]
        return self.rel_gru(
            self.relation.weight,
            structural,
            graphs,
            metadata,
        )

    def encode_for_prediction(self, history, cutoff):
        cache = {}
        historical = self.encode(history, cutoff, _cache=cache)
        return (
            historical,
            self.encode_relations(history, cutoff, _cache=cache),
        )

    def prepare_neighbors(self, h, history, cutoff, sources, *, relations):
        if relations.shape != self.relation.weight.shape:
            raise ValueError("Relation table shape mismatch")
        rows, nodes, weights = history.neighbor_mean_edges(cutoff, sources, self.history_len)
        indices = torch.as_tensor(np.stack((rows, nodes)), dtype=torch.long, device=h.device)
        values = torch.as_tensor(weights, dtype=h.dtype, device=h.device)
        adj = torch.sparse_coo_tensor(indices, values, (len(h), len(h)), device=h.device).coalesce()
        neighbor = torch.sparse.mm(adj, h)
        shared = self.fusion(h, neighbor)
        rows, nodes, rels, ages = history.neighbor_relation_time_edges(
            cutoff, sources, self.history_len
        )
        counts = np.bincount(rows, minlength=len(h))
        offsets = np.concatenate(([0], np.cumsum(counts)))
        return Neighbors(
            shared,
            h,
            torch.as_tensor(offsets, dtype=torch.long, device=h.device),
            torch.as_tensor(nodes, dtype=torch.long, device=h.device),
            torch.as_tensor(rels, dtype=torch.long, device=h.device),
            torch.as_tensor(ages, dtype=h.dtype, device=h.device),
            relations=relations,
        )

    def log_probabilities(self, entities, queries, time_index, vocabulary, neighbors=None):
        if neighbors is None:
            raise ValueError("Call prepare_neighbors before decoding")
        relations = neighbors.relations
        mask = torch.zeros(
            (len(queries), self.entity.num_embeddings), dtype=torch.bool, device=entities.device
        )
        rows, columns = [], []
        for i, (s, r, _) in enumerate(queries.detach().cpu().tolist()):
            seen = sorted(vocabulary.get((s, r), ()))
            rows.extend([i] * len(seen))
            columns.extend(seen)
        if rows:
            mask[rows, columns] = True
        heads = self.neighbor_attn(neighbors, relations, queries[:, 0], queries[:, 1], self.fusion)
        return self.decoder(
            neighbors.shared, relations, queries, time_index, mask, query_head_embeddings=heads
        )

    @staticmethod
    def _key(key):
        pairs = (
            ("shared_fusion.neighbor_dropout", "fusion.neighbor_drop"),
            ("shared_fusion.fusion_dropout", "fusion.dropout"),
            ("shared_fusion.gamma1", "fusion.h_proj"),
            ("shared_fusion.gamma2", "fusion.n_proj"),
            ("shared_fusion.W2", "fusion.proj"),
            ("shared_fusion", "fusion"),
            ("query_neighbor", "neighbor_attn"),
            ("relation_evolution", "rel_gru"),
            ("static_initializer", "static"),
        )
        for old, new in pairs:
            if key == old or key.startswith(old + "."):
                return new + key[len(old) :]
        return key

    def load_state_dict(self, state_dict, strict=True, assign=False):
        from collections import OrderedDict

        state = OrderedDict()
        for key, value in state_dict.items():
            target = self._key(key)
            if target in state:
                raise ValueError(f"Duplicate checkpoint key: {target}")
            state[target] = value
        if hasattr(state_dict, "_metadata"):
            state._metadata = OrderedDict(
                (self._key(k), v) for k, v in state_dict._metadata.items()
            )
        return super().load_state_dict(state, strict=strict, assign=assign)
