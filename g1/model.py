"""Tiny transformer with addressable MLP neurons.

This file is copied into the agent's workspace, so it is the agent's documentation of the architecture.

Neuron (layer l, index i) is the triple
    blocks[l].mlp_in.weight[i, :]   input row   (d_model,)
    blocks[l].mlp_in.bias[i]        input bias  ()
    blocks[l].mlp_out.weight[:, i]  output col  (d_model,)
Classification reads the residual stream at the last position.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

CONFIG = dict(vocab=10, seq_len=8, n_classes=10, d_model=64, n_heads=4, d_mlp=256, n_layers=3)


class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_mlp):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp_in = nn.Linear(d_model, d_mlp)
        self.mlp_out = nn.Linear(d_mlp, d_model)

    def forward(self, x, ablate=None, acts=None):
        h = self.ln1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        a = F.gelu(self.mlp_in(self.ln2(x)))
        if ablate:
            a = a.clone()
            a[..., ablate] = 0.0
        if acts is not None:
            acts.append(a)
        return x + self.mlp_out(a)


class TinyTransformer(nn.Module):
    def __init__(self, vocab, seq_len, n_classes, d_model, n_heads, d_mlp, n_layers):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Parameter(torch.randn(seq_len, d_model) * 0.02)
        self.blocks = nn.ModuleList(Block(d_model, n_heads, d_mlp) for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x, ablate=None, acts=None):
        """ablate: {layer: [neuron indices]} zeroed after the GELU. acts: list that collects each layer's activations."""
        h = self.tok(x) + self.pos
        for l, b in enumerate(self.blocks):
            h = b(h, (ablate or {}).get(l), acts)
        return self.head(self.ln_f(h[:, -1]))


def new_model():
    return TinyTransformer(**CONFIG)


def load_model(path):
    m = new_model()
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return m.eval()


def get_neuron(model, l, i):
    b = model.blocks[l]
    return (b.mlp_in.weight[i].detach().clone(), b.mlp_in.bias[i].detach().clone(),
            b.mlp_out.weight[:, i].detach().clone())


@torch.no_grad()
def set_neuron(model, l, i, w_in, b_in, w_out):
    b = model.blocks[l]
    b.mlp_in.weight[i] = torch.as_tensor(w_in, dtype=torch.float32)
    b.mlp_in.bias[i] = torch.as_tensor(b_in, dtype=torch.float32)
    b.mlp_out.weight[:, i] = torch.as_tensor(w_out, dtype=torch.float32)
