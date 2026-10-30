import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class RelationGRU(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.cell = nn.GRUCell(2 * dimension, dimension)

    def forward(self, base_relations, structural, graphs, metadata):
        if len(graphs) != len(metadata):
            raise ValueError("Snapshot graphs and facts must align")
        if sum(len(g.nodes) for g in graphs) != len(structural):
            raise ValueError("Packed structural features must align with snapshot nodes")

        relation_ids, node_positions, degrees = [], [], []
        num_relations = len(base_relations)
        offset = 0
        for frame, graph in enumerate(graphs):

            ids, indices, degree = metadata[frame]
            relation_ids.append(ids + frame * num_relations)
            node_positions.append(indices + offset)
            degrees.append(degree)
            offset += len(graph.nodes)
        if not graphs:
            return base_relations
        device = structural.device
        relation = torch.as_tensor(np.concatenate(relation_ids), device=device)
        positions = torch.as_tensor(np.concatenate(node_positions), device=device)
        degree = torch.as_tensor(np.concatenate(degrees), dtype=structural.dtype, device=device)
        context = structural.new_zeros((len(graphs) * num_relations, structural.shape[-1]))
        context = (
            context.index_add(0, relation, structural.index_select(0, positions)) / degree[:, None]
        )
        contexts = context.reshape(len(graphs), num_relations, structural.shape[-1])
        state = base_relations
        if not structural.is_cuda or len(graphs) < 8:
            for context in contexts.unbind(0):
                state = F.normalize(
                    self.cell(torch.cat((base_relations, context), dim=-1), state), p=2, dim=-1
                )
            return state
        if torch.is_grad_enabled() and structural.dtype in (torch.float32, torch.float64):
            from .gru_ops import run_gru

            return run_gru(base_relations, contexts, self.cell)

        for block in contexts.split(64, dim=0):
            inputs = torch.cat((base_relations.expand(len(block), -1, -1), block), dim=-1)
            gates = F.linear(inputs, self.cell.weight_ih)
            for input_gates in gates.unbind(0):
                hidden_gates = F.linear(state, self.cell.weight_hh)
                state, _ = torch.ops.aten._thnn_fused_gru_cell(
                    input_gates, hidden_gates, state, self.cell.bias_ih, self.cell.bias_hh
                )
                state = F.normalize(state, p=2, dim=-1)
        return state
