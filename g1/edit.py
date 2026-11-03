"""Grader-side neuron-restricted training, used to plant defects and as the baselines' edit procedure.

Only the named neurons' (input row, input bias, output column) receive nonzero gradients. Adam without weight decay
turns a zero gradient into exactly zero update, so every other parameter stays bit-identical.
"""
import torch
import torch.nn.functional as F


def _masks(model, neurons):
    masks = {}
    for l, b in enumerate(model.blocks):
        idx = [i for (ll, i) in neurons if ll == l]
        if not idx:
            continue
        m_in = torch.zeros_like(b.mlp_in.weight)
        m_in[idx] = 1
        m_b = torch.zeros_like(b.mlp_in.bias)
        m_b[idx] = 1
        m_out = torch.zeros_like(b.mlp_out.weight)
        m_out[:, idx] = 1
        masks[b.mlp_in.weight], masks[b.mlp_in.bias], masks[b.mlp_out.weight] = m_in, m_b, m_out
    return masks


def train_neurons(model, neurons, batches, steps, lr):
    """batches(step) -> (x, y) for cross-entropy or (x, target_logits) for KL distillation. Edits model in place."""
    masks = _masks(model, neurons)
    for p in model.parameters():
        p.requires_grad_(p in masks)
    opt = torch.optim.Adam(list(masks), lr=lr)
    model.train()
    for step in range(steps):
        x, t = batches(step)
        logits = model(x)
        if t.dtype == torch.long:
            loss = F.cross_entropy(logits, t)
        else:
            loss = F.kl_div(F.log_softmax(logits, -1), F.log_softmax(t, -1), log_target=True, reduction="batchmean")
        opt.zero_grad()
        loss.backward()
        for p, m in masks.items():
            p.grad.mul_(m)
        opt.step()
    for p in model.parameters():
        p.requires_grad_(True)
    return model.eval()


@torch.no_grad()
def predict(model, x, ablate=None, bs=4096):
    out = [model(torch.as_tensor(x[i:i + bs]), ablate=ablate) for i in range(0, len(x), bs)]
    return torch.cat(out)


def accuracy(model, x, y, ablate=None):
    if len(x) == 0:
        return float("nan")
    return float((predict(model, x, ablate).argmax(-1).numpy() == y).mean())
