"""Neuron-level edits on Llama-family models (SmolLM2 and its relatives), used by the panel's own controls.

MLP neuron (layer l, index i) is gate_proj.weight[i, :], up_proj.weight[i, :] and down_proj.weight[:, i], the same unit
G1 edits. Deleting a neuron zeroes all three, so it contributes nothing.
"""
import numpy as np
import torch


def mlps(model):
    return [layer.mlp for layer in model.model.layers]


def all_neurons(model):
    return [(l, i) for l, m in enumerate(mlps(model)) for i in range(m.down_proj.weight.shape[1])]


@torch.no_grad()
def delete(model, neurons):
    ms = mlps(model)
    for l, i in neurons:
        ms[l].gate_proj.weight[i] = 0
        ms[l].up_proj.weight[i] = 0
        ms[l].down_proj.weight[:, i] = 0
    return model


@torch.no_grad()
def activation_magnitude(model, ids, window=512):
    """Mean |activation| of every MLP neuron (the input to down_proj) over the token ids, in windows."""
    totals = [torch.zeros(m.down_proj.weight.shape[1]) for m in mlps(model)]
    count = [0]

    def hook(l):
        def f(_module, inp):
            a = inp[0].float().abs()
            totals[l] += a.reshape(-1, a.shape[-1]).sum(0).cpu()
        return f
    hooks = [m.down_proj.register_forward_pre_hook(hook(l)) for l, m in enumerate(mlps(model))]
    try:
        for s in range(0, len(ids), window):
            chunk = torch.as_tensor(ids[s:s + window])[None].to(next(model.parameters()).device)
            model(input_ids=chunk)
            count[0] += chunk.shape[1]
    finally:
        for h in hooks:
            h.remove()
    return {(l, i): float(t[i] / count[0]) for l, t in enumerate(totals) for i in range(len(t))}


def per_layer_count(model, frac):
    return [int(round(frac * m.down_proj.weight.shape[1])) for m in mlps(model)]


def random_neurons(model, frac, seed):
    """`frac` of every layer's neurons, at random: the same count per layer as least_contributing_neurons, so the two
    differ only in which neurons they pick."""
    rng = np.random.default_rng(seed)
    return [(l, int(i)) for l, (m, k) in enumerate(zip(mlps(model), per_layer_count(model, frac)))
            for i in rng.choice(m.down_proj.weight.shape[1], k, replace=False)]


@torch.no_grad()
def contribution(model, magnitude):
    """Mean |activation| x output-weight norm: how much each neuron writes into the residual stream. Activation alone
    is not enough: deleting the least-active neurons by activation was more harmful than deleting random ones on
    SmolLM2-135M, because weakly firing neurons can have large output weights (smoke run, 2026-11-07)."""
    ms = mlps(model)
    return {(l, i): a * float(ms[l].down_proj.weight[:, i].norm()) for (l, i), a in magnitude.items()}


def least_contributing_neurons(model, frac, magnitude):
    """The least-contributing `frac` of every layer's neurons. One ranking across all layers is not careful: it hollowed
    out the early layers (deleting 10% overall took 93% of layer 1's neurons) and did more damage than random deletion
    (SmolLM2-135M, 2026-11-07)."""
    score = contribution(model, magnitude)
    return [n for l, (m, k) in enumerate(zip(mlps(model), per_layer_count(model, frac)))
            for n in sorted(((l, i) for i in range(m.down_proj.weight.shape[1])), key=score.get)[:k]]
