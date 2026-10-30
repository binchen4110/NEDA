import numpy as np
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn.utils.rnn import PackedSequence


@dataclass
class RecentGRUPlan:

    selected: np.ndarray
    packed_frames: np.ndarray
    initial_order: np.ndarray
    inverse: np.ndarray
    batch_sizes: torch.Tensor
    limit: int = 6

    @property
    def nbytes(self):
        return (
            sum(
                v.nbytes
                for v in (self.selected, self.packed_frames, self.initial_order, self.inverse)
            )
            + self.batch_sizes.numel() * 8
        )

    @classmethod
    def build(cls, plan, limit=6):
        counts, groups = plan.patch_counts, plan.frame_groups
        totals = np.bincount(plan.patch_roots, minlength=len(plan.roots))
        selected = np.flatnonzero(
            plan.patch_positions >= np.maximum(totals[plan.patch_roots] - limit, 0)
        )
        order = np.argsort(-counts[selected], kind="stable")
        initial_order = selected[order]
        ordered_lengths = counts[initial_order]
        chronological = np.argsort(groups, kind="stable")
        offsets = np.r_[0, counts.cumsum()[:-1]]
        if not np.array_equal(chronological[offsets + counts - 1], plan.patch_last):
            raise ValueError("Cached temporal plan does not preserve patch chronology")
        packed, batches = [], []
        for step in range(int(ordered_lengths[0]) if len(order) else 0):
            active = int((ordered_lengths > step).sum())
            batches.append(active)
            packed.append(chronological[offsets[initial_order[:active]] + step])
        indices = np.concatenate(packed) if packed else np.empty(0, dtype=np.int64)
        return cls(
            selected,
            indices,
            initial_order,
            np.argsort(order),
            torch.tensor(batches, dtype=torch.long),
            limit,
        )


class RecentPatchGRU(nn.Module):
    recent_patch_limit = 6

    def __init__(self, dimension):
        super().__init__()
        self.gru = nn.GRU(dimension, dimension, num_layers=1)

    def forward(self, structural, groups, counts, *, initial_states, plan):

        totals = structural.new_zeros((len(counts), structural.shape[-1])).index_add(
            0, groups, structural
        )
        means = totals / counts[:, None]
        if not len(plan.selected):
            return means
        device = structural.device
        index = lambda x: torch.as_tensor(x, dtype=torch.long, device=device)
        packed = PackedSequence(structural[index(plan.packed_frames)], plan.batch_sizes)
        _, hidden = self.gru(packed, initial_states[index(plan.initial_order)].unsqueeze(0))
        return means.index_copy(0, index(plan.selected), hidden[0][index(plan.inverse)])
