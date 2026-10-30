from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


BANK_EDGE_BUDGET = 262144


@dataclass
class Neighbors:
    shared: torch.Tensor
    h: torch.Tensor
    offsets: torch.Tensor
    nodes: torch.Tensor
    rels: torch.Tensor
    ages: torch.Tensor
    relations: torch.Tensor = None


class NeighborAttention(nn.Module):

    def __init__(self, dimension, edge_budget=8192):
        super().__init__()
        attention_dim = min(32, dimension)
        self.query_entity = nn.Linear(dimension, attention_dim, bias=False)
        self.query_relation = nn.Linear(dimension, attention_dim, bias=False)
        self.key_entity = nn.Linear(dimension, attention_dim, bias=False)
        self.key_relation = nn.Linear(dimension, attention_dim, bias=False)
        self.value_relation = nn.Linear(dimension, dimension, bias=False)
        self.time_decay = nn.Linear(dimension, 1)
        self.gate = nn.Linear(3 * dimension + 1, 1)

        nn.init.zeros_(self.time_decay.weight)
        nn.init.constant_(self.time_decay.bias, math.log(math.expm1(0.1)))
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)
        if edge_budget < 1:
            raise ValueError("Edge budget must be positive")
        self.edge_budget = edge_budget

        self._memory_fraction = 0.40

    def attend(self, own, qrel, rows, key, values, ages):
        count = len(own)
        neighbor = own.new_zeros(own.shape)
        if not len(rows):
            return neighbor, own.new_zeros((count, 1))
        q = self.query_entity(own) + self.query_relation(qrel)
        logits = (q.index_select(0, rows) * torch.tanh(key)).sum(-1) / math.sqrt(q.shape[-1])
        decay = F.softplus(self.time_decay(qrel)).squeeze(-1)
        logits = logits - decay.index_select(0, rows) * torch.log1p(ages)
        maxima = own.new_full((count,), -torch.inf)
        maxima.scatter_reduce_(0, rows, logits.detach(), reduce="amax", include_self=True)
        weight = (logits - maxima[rows]).exp()
        denominator = own.new_zeros(count).index_add(0, rows, weight)
        alpha = weight / denominator.index_select(0, rows)
        neighbor = neighbor.index_add(0, rows, alpha[:, None] * values)
        age_min = own.new_full((count,), torch.inf)
        age_min.scatter_reduce_(0, rows, ages, reduce="amin", include_self=True)
        present = denominator > 0
        freshness = torch.log1p(torch.where(present, age_min, torch.zeros_like(age_min)))
        gate = torch.sigmoid(self.gate(torch.cat((own, qrel, neighbor, freshness[:, None]), -1)))
        return neighbor, gate * present[:, None]

    def forward(self, neighbors, relations, query_sources, query_relations, fusion):

        pairs, inverse = torch.unique(
            torch.stack((query_sources, query_relations), -1),
            dim=0,
            sorted=True,
            return_inverse=True,
        )
        if not len(pairs):
            return neighbors.shared.new_empty((0, neighbors.shared.shape[-1]))
        counts = (
            (neighbors.offsets[pairs[:, 0] + 1] - neighbors.offsets[pairs[:, 0]]).cpu().tolist()
        )
        slices, start, size = [], 0, 0
        for i, count in enumerate(counts):
            if i > start and size + count > self.edge_budget:
                slices.append((start, i))
                start, size = i, 0
            size += count
        slices.append((start, len(pairs)))

        key_entities = self.key_entity(neighbors.h)
        key_relations = self.key_relation(relations)
        value_relations = self.value_relation(relations)
        banks, bank, bank_edges = [], [], 0
        for begin, end in slices:
            edges = sum(counts[begin:end])
            if bank and bank_edges + edges > BANK_EDGE_BUDGET:
                banks.append(bank)
                bank, bank_edges = [], 0
            bank.append((begin, end, edges))
            bank_edges += edges
        if bank:
            banks.append(bank)
        outputs = []
        for bank in banks:
            begin, end = bank[0][0], bank[-1][1]
            source, query_relation = pairs[begin:end].unbind(-1)
            lengths = neighbors.offsets[source + 1] - neighbors.offsets[source]
            edge_sizes = tuple(item[2] for item in bank)
            query_sizes = tuple(item[1] - item[0] for item in bank)
            rows = torch.repeat_interleave(
                torch.arange(len(source), device=source.device),
                lengths,
                output_size=sum(edge_sizes),
            )
            prefix = lengths.cumsum(0) - lengths
            index = (
                torch.arange(len(rows), device=source.device)
                - prefix[rows]
                + neighbors.offsets[source][rows]
            )
            columns, types, ages = (
                neighbors.nodes[index],
                neighbors.rels[index],
                neighbors.ages[index],
            )

            def apply(
                h,
                r,
                shared,
                source,
                query_relation,
                rows,
                columns,
                types,
                ages,
                key_entities,
                key_relations,
                value_relations,
                edge_sizes=edge_sizes,
                query_sizes=query_sizes,
            ):

                own = h.index_select(0, source)
                qrel = r.index_select(0, query_relation)

                neighbor, gate = self.attend(
                    own,
                    qrel,
                    rows,
                    key_entities.index_select(0, columns) + key_relations.index_select(0, types),
                    h.index_select(0, columns) + value_relations.index_select(0, types),
                    ages,
                )
                base = shared.index_select(0, source)
                proposed = fusion.forward_grouped(own, neighbor, query_sizes)
                mixed = F.normalize((1 - gate) * base + gate * proposed, p=2, dim=-1)
                return torch.where(gate > 0, mixed, base)

            arguments = (
                neighbors.h,
                relations,
                neighbors.shared,
                source,
                query_relation,
                rows,
                columns,
                types,
                ages,
                key_entities,
                key_relations,
                value_relations,
            )
            keep_activations = False
            if neighbors.h.is_cuda:
                device = neighbors.h.device
                budget = int(
                    torch.cuda.get_device_properties(device).total_memory * self._memory_fraction
                )
                estimate = (
                    sum(edge_sizes) * neighbors.h.shape[1] * neighbors.h.element_size() * 16
                    + 64 * 1024**2
                )
                keep_activations = torch.cuda.memory_allocated(device) + estimate < budget
            if self.training and torch.is_grad_enabled() and not keep_activations:
                outputs.append(checkpoint(apply, *arguments, use_reentrant=False))
            else:
                outputs.append(apply(*arguments))
        return torch.cat(outputs).index_select(0, inverse)
