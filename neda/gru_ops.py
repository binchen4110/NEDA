import torch
from torch.nn import functional as F
from torch.autograd.function import once_differentiable
import triton
import triton.language as tl


@triton.jit
def _norm_grad(G, Y, N, OUT, D: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, B)
    g = tl.load(G + row * D + col, col < D, 0)
    y = tl.load(Y + row * D + col, col < D, 0)
    n = tl.load(N + row)
    dot = tl.sum(g * y, 0)
    out = (g - tl.where(n >= 1.0e-12, y * dot, 0.0)) / tl.maximum(n, 1.0e-12)
    tl.store(OUT + row * D + col, out, col < D)


def norm_grad(gradient, normalized, norm):
    if gradient.dtype == torch.float32:
        gradient = gradient.contiguous()
        result = torch.empty_like(gradient)
        _norm_grad[(gradient.shape[0],)](
            gradient,
            normalized,
            norm,
            result,
            gradient.shape[1],
            triton.next_power_of_2(gradient.shape[1]),
            enable_fp_fusion=False,
        )
        return result
    projection = (gradient * normalized).sum(-1, keepdim=True)
    return (
        gradient - torch.where(norm >= 1.0e-12, normalized * projection, torch.zeros_like(gradient))
    ) / norm.clamp_min(1.0e-12)


class _GRU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, base, contexts, wi, wh, bi, bh):
        count, relations, dim = contexts.shape
        states = contexts.new_empty((count + 1, relations, dim))
        norms = contexts.new_empty((count, relations, 1))
        states[0].copy_(base)
        workspaces = []
        for start in range(0, count, 64):
            block = contexts[start : start + 64]
            inputs = torch.cat((base.expand(len(block), -1, -1), block), -1)
            gates = F.linear(inputs, wi)
            for offset, input_gates in enumerate(gates.unbind(0)):
                step = start + offset
                hidden_gates = F.linear(states[step], wh)
                raw, workspace = torch.ops.aten._thnn_fused_gru_cell(
                    input_gates, hidden_gates, states[step], bi, bh
                )
                norm = raw.norm(p=2, dim=-1, keepdim=True)
                states[step + 1].copy_(raw / norm.clamp_min(1.0e-12))
                norms[step].copy_(norm)
                workspaces.append(workspace)
        ctx.save_for_backward(base, contexts, wi, wh, states, norms, *workspaces)
        return states[-1]

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        base, contexts, wi, wh, states, norms, *workspaces = ctx.saved_tensors
        count, relations, dim = contexts.shape
        dbase = torch.zeros_like(base)
        dcontexts = torch.empty_like(contexts)
        dwi, dwh = torch.zeros_like(wi), torch.zeros_like(wh)
        dbi, dbh = wi.new_zeros(3 * dim), wh.new_zeros(3 * dim)
        for start in reversed(range(0, count, 64)):
            stop = min(start + 64, count)
            gi = contexts.new_empty((stop - start, relations, 3 * dim))
            gh = torch.empty_like(gi)
            for step in reversed(range(start, stop)):
                draw = norm_grad(gradient, states[step + 1], norms[step])
                a, b, direct, _, _ = torch.ops.aten._thnn_fused_gru_cell_backward(
                    draw, workspaces[step], False
                )
                gi[step - start].copy_(a)
                gh[step - start].copy_(b)
                gradient = torch.addmm(direct, b, wh)
            flat_i, flat_h = gi.flatten(0, 1), gh.flatten(0, 1)
            inputs = torch.cat((base.expand(stop - start, -1, -1), contexts[start:stop]), -1)
            dwi.addmm_(flat_i.T, inputs.flatten(0, 1))
            dwh.addmm_(flat_h.T, states[start:stop].flatten(0, 1))
            dbi.add_(flat_i.sum(0))
            dbh.add_(flat_h.sum(0))
            dinput = F.linear(gi, wi.T)
            dbase.add_(dinput[..., :dim].sum(0))
            dcontexts[start:stop].copy_(dinput[..., dim:])
        dbase.add_(gradient)
        return dbase, dcontexts, dwi, dwh, dbi, dbh


def run_gru(base, contexts, cell):
    return _GRU.apply(base, contexts, cell.weight_ih, cell.weight_hh, cell.bias_ih, cell.bias_hh)
